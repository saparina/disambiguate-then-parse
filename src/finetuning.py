import copy
import os
import json
import shutil

import yaml

import wandb
import configargparse

from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

import random
import numpy as np

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

from utils import (
    EvaluationMetricsTracker, 
    evaluate_predicted_statements, 
    parse_statements_llama, 
    init_model, 
    generate_from_prompt,
    get_generation_config
)
from dataset import load_finetuning_datasets
from prompts import (
    user_message_interpretations, 
    user_message_sql_ambig, 
    user_message_sql, 
    user_message_interpr_ambig_missing
)
 
def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# Log arguments for reproducibility
def log_arguments(args, log_file="args.json"):
    with open(log_file, "w") as f:
        json.dump(vars(args), f, indent=4)

def add_script_arguments(parser):
    parser.add_argument("--config", is_config_file=True, help="Path to YAML configuration file.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite existing output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exp_name", type=str, help="Name of the experiment.")
    parser.add_argument("--output_dir", type=str, default="outputs/finetune", help="Directory to save outputs.")
    parser.add_argument("--mode", type=str, choices=["validation", "test", "all", "validation,test"], default="all", help="Execution mode. Can select one or two modes using comma separator.")
    parser.add_argument("--test_checkpoint", type=str, help="Path to test checkpoint.")

    parser.add_argument("--learn_missing_interpr", action="store_true", help="Learn missing interpretations.")
    parser.add_argument("--learn_gold_interpr", action="store_true", help="Learn gold interpretations.")

    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--validation_checkpoints", type=int, default=3, help="Number of validation checkpoints.")
    parser.add_argument("--num_keep_checkpoints", type=int, default=1, help="Number of checkpoints to keep.")
    parser.add_argument("--final_metric", type=str, default="recall", help="Final metric to evaluate.")

    parser.add_argument("--remove_duplicates_predictions", action="store_true", help="Remove duplicate predictions in metric.")
    parser.add_argument("--skip_sql_test", action="store_true", help="Skip SQL prediction during testing for interpretation models")
    parser.add_argument("--skip_validation", action="store_true", help="Skip validation during training")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from checkpoint")

def add_model_arguments(parser):
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model.")
    parser.add_argument("--model_sql_name", type=str, required=True, help="Name of the SQL-specific model.")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token")
    parser.add_argument("--chat_template", type=str, help="Chat template for main model")
    parser.add_argument("--sql_chat_template", type=str, help="Chat template for SQL model")
    parser.add_argument("--max_seq_length", type=int, default=8192, help="Maximum sequence length.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization for models.")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "auto"], default="auto")

    # Generation config parameters
    parser.add_argument("--max_new_tokens", type=int, default=2000, help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")


def add_training_arguments(parser):
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimization.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warm-up ratio for the scheduler.")
    parser.add_argument("--save_steps", type=int, default=500, help="Steps between saving checkpoints.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Steps between logging.")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load the best model at the end.")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting tool (e.g., wandb).")
    parser.add_argument("--eval_steps", type=int, default=None, help="Steps between evaluations")
    parser.add_argument("--eval_strategy", type=str, default="no", choices=["no", "steps", "epoch"], help="Evaluation strategy")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--auto_find_batch_size", action="store_true", help="Automatically find batch size")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--group_by_length", action="store_true", help="Group sequences of similar length")
    parser.add_argument("--neftune_noise_alpha", type=float, default=None, help="NEFTune noise alpha")
    parser.add_argument("--batch_eval_metrics", action="store_true", help="Evaluate metrics in batches")
    parser.add_argument("--interpretation_model_train", type=str, default=None, help="Interpretation model to use for training")
    parser.add_argument("--interpretation_model_test", type=str, default=None, help="Interpretation model to use for test")

def add_lora_arguments(parser):
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank for fine-tuning.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha for fine-tuning.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout for fine-tuning.")


def add_dataset_arguments(parser):
    """Add dataset-related arguments to parser"""
    parser.add_argument("--dataset_type", type=str, choices=["ambrosia", "ambiqt", "all"], default="all", help="Type of dataset.")
    parser.add_argument("--dataset_type_test", type=str, choices=["ambrosia", "ambiqt", "all"], default="all", help="Type of test dataset.")
    parser.add_argument("--ambrosia_file", type=str, default="data/ambrosia/data/ambrosia_resplit.csv", help="Path to Ambrosia dataset file")
    parser.add_argument("--data_dir", type=str, default="data/", help="Path to AmbiQT dataset directory")
    parser.add_argument("--ambiqt_interpr_file", type=str, default=None, help="Path to AmbiQT gold interpretations")
    parser.add_argument("--question_type", type=str, default=None, choices=[None, "ambig", "unambig"], help="Type of questions to use")
    parser.add_argument("--question_type_test", type=str, default=None, choices=[None, "ambig", "unambig"], help="Type of questions to use for test")
    parser.add_argument("--balance_dataset", action="store_true", help="Whether to balance dataset")
    parser.add_argument("--sql_output_dir", type=str, default="outputs/sql_generation_filtered", help="Path to initial interpretaions with SQL queries.")

# Parse arguments dynamically from YAML
def parse_args():
    parser = configargparse.ArgParser(default_config_files=["src/configs/train.yaml"])

    add_script_arguments(parser)
    add_model_arguments(parser)
    add_training_arguments(parser)
    add_lora_arguments(parser)
    add_dataset_arguments(parser)  # Add dataset arguments

    with open("src/configs/train.yaml", "r") as f:
        yaml_config = yaml.safe_load(f)
    
    # Collect existing arguments to avoid conflicts
    existing_args = {action.option_strings[0] for action in parser._actions if action.option_strings}

    for key, value in yaml_config.items():
        arg_name = f"--{key}"
        if arg_name in existing_args:
            continue  # Skip if the argument already exists
        if isinstance(value, bool):
            parser.add_argument(arg_name, action="store_true" if value else "store_false")
        elif isinstance(value, (int, float, str)):
            parser.add_argument(arg_name, type=type(value), default=value)
        else:
            parser.add_argument(arg_name, default=value)

    args = parser.parse_args()
    return args


def train(args, train_dataset, val_dataset):
    model_wrapper, tokenizer = init_model(args, for_inference=False)
    
    model = FastLanguageModel.get_peft_model(
        model_wrapper.model,
        r = args.lora_r,
        target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head"
        ],
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
        use_rslora = False,
        loftq_config = None,
    )

    peft_config = None

    fp16 = not is_bfloat16_supported()
    bf16 = is_bfloat16_supported()
    optim = "adamw_8bit"

    training_arguments = SFTConfig(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=args.num_epochs,
        load_best_model_at_end=args.load_best_model_at_end,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps = 1,
        max_grad_norm = args.max_grad_norm,
        auto_find_batch_size = args.auto_find_batch_size,
        save_total_limit = args.validation_checkpoints,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        fp16=fp16,
        bf16=bf16,
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to=args.report_to,
        neftune_noise_alpha= args.neftune_noise_alpha,
        optim=optim,
        batch_eval_metrics=args.batch_eval_metrics,
        do_eval = False,
        save_strategy="epoch",
    )

    packing = False
    training_arguments.packing = False


    def formatting_prompts_func(examples):
        db_dumps = examples["db_dump"]
        inputs = examples["question"]

        if args.learn_gold_interpr:
            user_message = user_message_interpretations
            outputs      = examples["nl_interpretations"]
        elif args.learn_missing_interpr:
            user_message = user_message_interpr_ambig_missing
            outputs      = examples["missing_nl_interpretations"]
        else:
            user_message = user_message_sql_ambig
            outputs      = examples["gold_queries"]
        

        if isinstance(inputs, list):
            output_texts = []
            for idx, (db_dump, input, output) in enumerate(zip(db_dumps, inputs, outputs)):
                if args.learn_missing_interpr:
                    answer = output
                elif args.learn_gold_interpr:
                    answer = output.replace("\n\n", "\n")
                else:
                    answer = "\n\n".join(output)

                if args.learn_missing_interpr:
                    messages = [
                        {"role": "user", "content": user_message.format(db_dump, input, "\n".join(examples["initial_generated_interpr"][idx]))},
                        {"role": "assistant", "content": answer},
                    ]
                else:
                    messages = [
                        {"role": "user", "content": user_message.format(db_dump, input)},
                        {"role": "assistant", "content": answer},
                    ]
                text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)
                
                output_texts.append(text)
        elif isinstance(inputs, str):
            output_texts = []
            if args.learn_gold_interpr or args.learn_missing_interpr:
                answer = outputs
            else:
                answer = "\n\n".join(outputs)

            if args.learn_missing_interpr:
                messages = [
                    {"role": "user", "content": user_message.format(db_dumps, inputs, "\n".join(examples["initial_generated_interpr"]))},
                    {"role": "assistant", "content": answer},
                ]
            else:
                messages = [
                        {"role": "user", "content": user_message.format(db_dumps, inputs)},
                        {"role": "assistant", "content": answer},
                ]
            text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)
            output_texts.append(text)
        else:
            import pdb; pdb.set_trace()

        for text in output_texts:
            if "no interpretations" in text.lower():
                import pdb; pdb.set_trace()


        return output_texts

    log_arguments(args)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,  # Use original dataset
        eval_dataset=val_dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        tokenizer=tokenizer,
        packing=packing,
        args=training_arguments,
        max_seq_length=args.max_seq_length,
        dataset_num_proc=1,
    )

    if "llama" in args.model_name.lower():
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "qwen" in args.model_name.lower():
        instruction_part = "<|im_start|>user\n"
        response_part = "<|im_start|>assistant\n"
    elif "gemma" in args.model_name.lower():
        instruction_part = "<bos><start_of_turn>user\n"
        response_part = "<start_of_turn>model\n"
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    trainer = train_on_responses_only(
        trainer,
        instruction_part = instruction_part,
        response_part = response_part,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

def inference(model, eval_dataset, res_path, user_message=None, skip_sql=False):
    model, tokenizer = init_model(args, model_name=model, for_inference=True)
    generation_config = get_generation_config(args, model)
    del generation_config["max_length"]

    if (args.learn_gold_interpr or args.learn_missing_interpr) and not skip_sql:
        model_sql, tokenizer_sql = init_model(args, use_sql_model=True, for_inference=True)
        generation_config_sql = get_generation_config(args, model_sql)
        del generation_config_sql["max_length"]
        print(model_sql.model)

    # Initialize metrics tracker
    metrics_tracker = EvaluationMetricsTracker()
    results = []

    message_sql = user_message if user_message else user_message_sql
    
    if args.learn_gold_interpr:
        user_message = user_message_interpretations
    elif args.learn_missing_interpr:
        user_message = user_message_interpr_ambig_missing
    else:
        user_message = user_message_sql_ambig

    for example in tqdm(eval_dataset):
        db_dump = example["db_dump"]
        question = example["question"]

        initial_generated_interpr = example.get("initial_generated_interpr", [])
        assert initial_generated_interpr is not None or not args.learn_gold_interpr or not args.learn_missing_interpr, "initial_generated_interpr is None"

        if args.learn_missing_interpr:
            messages = [
                {"role": "user", "content": user_message.format(db_dump, question, "\n".join(initial_generated_interpr))}
            ]
        else:
            messages = [
                {"role": "user", "content": user_message.format(db_dump, question)},
            ]

        ambig_type = example["ambig_type"]
        main_key = "ambig" if example["is_ambiguous"] else "unambig"
        gold_queries = example['gold_queries']

        local_results = {
            "db_file": example['db_file'],
            "db_dump": example["db_dump"],
            "question": example["question"],
            "gold_queries": gold_queries,
            "is_ambiguous": example["is_ambiguous"], 
            "ambig_type": example["ambig_type"]
        }

        predictions = generate_from_prompt(
            model_wrapper=model,
            tokenizer=tokenizer,
            messages=messages,
            generation_config=generation_config
        )

        if args.learn_gold_interpr or args.learn_missing_interpr:
            predicted_interpretations = predictions.split("\n") if predictions is not None else []

            if args.learn_gold_interpr:
                local_results["all_interpretations"] = predicted_interpretations
            else:
                local_results["all_interpretations"] = initial_generated_interpr + predicted_interpretations

            local_results["initial_generated_interpr"] = initial_generated_interpr
            local_results["predicted_interpr"] = predicted_interpretations
        else:
            parsed_final_sql_queries = parse_statements_llama(predictions)
    

        if (args.learn_gold_interpr or args.learn_missing_interpr) and skip_sql:
            local_results["initial_generated_interpr_sql"] = {interp["interpretation"]: interp["sql_queries"] for interp in example["initial_generated_interpr"]}

            if args.learn_missing_interpr:
                local_results["all_interpretations_sql"] = local_results["initial_generated_interpr_sql"]
            else:
                local_results["all_interpretations_sql"] = {}

            local_results["generated_interpretations_sql"] = {interpr: None for interpr in predicted_interpretations}
            local_results["all_interpretations_sql"].update(local_results["generated_interpretations_sql"])
            
            local_results["predicted_interpr_sql"] = None
            local_results["sql_predictions"] = None
            results.append(local_results)
            continue


        if args.learn_gold_interpr or args.learn_missing_interpr:
            local_results["all_interpretations_sql"] = {}
            all_sql_queries = []
            interp_pred = {}

            def get_sql_queries(interpr):
                sql_messages = [
                    {"role": "user", "content": message_sql.format(example["db_dump"], interpr)},
                ]
                
                sql_predictions_from_interp = generate_from_prompt(
                    model_wrapper=model_sql,
                    tokenizer=tokenizer_sql,
                    messages=sql_messages,
                    generation_config=generation_config_sql
                )
                
                return parse_statements_llama(sql_predictions_from_interp) if sql_predictions_from_interp else ""

            if args.learn_missing_interpr:
                for interpr_details in example["generated_interpretations"]:
                    interpr = interpr_details["interpretation"]
                    if interpr.strip().lower().startswith("all possible interpretations are covered"):
                        print("All possible interpretations are covered while learning gold/corrected interpretations")
                        import pdb; pdb.set_trace()
                    if interpr.strip().lower().startswith("no interpretations"):
                        print("No interpretations while learning gold/corrected interpretations")
                        import pdb; pdb.set_trace()

                    # parsed_sql_predictions_from_interp = get_sql_queries(interpr)
                    parsed_sql_predictions_from_interp = interpr_details["sql_queries"]
                    interp_pred[interpr] = parsed_sql_predictions_from_interp
                    all_sql_queries += parsed_sql_predictions_from_interp

                local_results["generated_interpretations_sql"] = interp_pred
                local_results["all_interpretations_sql"] = copy.deepcopy(local_results["generated_interpretations_sql"])

            interp_pred = {}
            for interpr in local_results["predicted_interpr"]:
                if interpr.strip().lower().startswith("all possible interpretations are covered"):
                    if args.learn_gold_interpr:
                        print("All possible interpretations are covered while learning gold/corrected interpretations")
                        import pdb; pdb.set_trace()
                    continue
                if interpr.strip().lower().startswith("no interpretations"):
                    print("No interpretations while learning gold/corrected interpretations")
                    import pdb; pdb.set_trace()

                parsed_sql_predictions_from_interp = get_sql_queries(interpr)
                interp_pred[interpr] = parsed_sql_predictions_from_interp
                all_sql_queries += parsed_sql_predictions_from_interp

            local_results["predicted_interpr_sql"] = interp_pred
            local_results["all_interpretations_sql"].update(local_results["predicted_interpr_sql"])
            parsed_final_sql_queries = all_sql_queries

        local_results["sql_predictions"] = parsed_final_sql_queries

        if args.learn_missing_interpr or args.learn_gold_interpr:
            parsed_final_sql_queries = list(set(parsed_final_sql_queries))
        
        if parsed_final_sql_queries is None:
            metrics_tracker.add_zero_metrics(main_key, ambig_type)

            local_results.update({
                'precision': 0,
                'recall': 0,
                'all_found': 0,
                'one_found': 0,
                'f1_score': 0
            })

            results.append(local_results)
            continue

        try:
            metrics = evaluate_predicted_statements(
                example['db_file'], 
                parsed_final_sql_queries, 
                gold_queries, 
                remove_duplicates_predictions=args.remove_duplicates_predictions, 
                verbose=False
            )

            # Update metrics using the tracker
            metrics_tracker.update_metrics(main_key, ambig_type, metrics)

            # Add metrics to local results
            local_results.update(metrics_tracker.get_result_metrics(metrics))

        except:
            # Add zero metrics for failed predictions
            metrics_tracker.add_zero_metrics(main_key, ambig_type)
            
            local_results.update({
                'precision': 0,
                'recall': 0,
                'all_found': 0,
                'one_found': 0,
                'f1_score': 0
            })

        results.append(local_results)

    # Get aggregated metrics
    aggregated_metrics = metrics_tracker.get_aggregated_metrics()

    # Save results and metrics
    json.dump({
        "metrics": aggregated_metrics,
        "all_results": results,
    }, open(res_path, "w"), indent=4)
    
    # Log metrics to wandb
    recall_ambig_total = aggregated_metrics["ambig"]["total"]["recall"]
    wandb.log({"recall_ambig_total": recall_ambig_total})

    # Print metrics summary
    metrics_tracker.print_summary()

    return aggregated_metrics

def validation(args, eval_dataset):
    checkpoint_metric_list = []
    # Filter and sort checkpoints numerically
    checkpoints = []
    for checkpoint_dir in os.listdir(args.output_dir):
        if checkpoint_dir.startswith('checkpoint-'):
            try:
                checkpoint_num = int(checkpoint_dir.split('-')[1])
                checkpoints.append((checkpoint_num, checkpoint_dir))
            except (IndexError, ValueError):
                continue
    
    # Sort checkpoints by number in descending order
    checkpoints.sort(reverse=True)
    
    for _, checkpoint_dir in checkpoints:
        checkpoint_path = os.path.join(args.output_dir, checkpoint_dir)
        file_name = "predictions"
        file_name += f"_{args.exp_name}" if args.exp_name else ""
        file_name += f"_{args.question_type}" if args.question_type else ""
        file_name += f"_interpr" if args.learn_gold_interpr else ""
        file_name += f"_learn_missing_interpr" if args.learn_missing_interpr else ""
        file_name += f"_learn_gold_interpr" if args.learn_gold_interpr else ""
        file_name += f"_epochs{args.num_epochs}" if args.num_epochs else ""
        file_name += f"_seed{args.seed}" if args.seed else ""
        file_name += f"_validation"
        file_name = os.path.join(checkpoint_path, file_name)
        
        try:
            sum_metrics = inference(checkpoint_path, eval_dataset, res_path=file_name)
            checkpoint_metric_list.append((np.mean(sum_metrics["ambig"]["total"][args.final_metric]), checkpoint_path))
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint_path}: {str(e)}")
            continue

    checkpoint_metric_list.sort(key=lambda x: x[0], reverse=True)
    top_k_checkpoints = checkpoint_metric_list[:args.num_keep_checkpoints]

    for _, checkpoint_path in checkpoint_metric_list[args.num_keep_checkpoints:]:
        print(f"Deleting checkpoint: {checkpoint_path}")
        shutil.rmtree(checkpoint_path)

    print(f"Top {args.num_keep_checkpoints} checkpoints kept:")
    for metric, path in top_k_checkpoints:
        print(f"Checkpoint: {path} with {args.final_metric}: {metric}")
    return top_k_checkpoints


def test(args, checkpoint_path, test_dataset, dataset_type="ambrosia", user_message=None):
    file_name = "predictions"
    file_name += f"_{args.exp_name}" if args.exp_name else ""
    file_name += f"_{args.question_type}" if args.question_type else ""
    file_name += f"_interpr" if args.learn_gold_interpr else ""
    file_name += f"_learn_missing_interpr" if args.learn_missing_interpr else ""
    file_name += f"_learn_gold_interpr" if args.learn_gold_interpr else ""
    file_name += f"_epochs{args.num_epochs}" if args.num_epochs else ""
    file_name += f"_seed{args.seed}" if args.seed else ""
    file_name += f"_interpret_model_{args.interpretation_model_test}" if args.interpretation_model_test else ""
    file_name += f"_test_{dataset_type}"
    file_name = os.path.join(checkpoint_path, file_name)

    # Add skip_sql parameter to inference call
    skip_sql = (args.learn_gold_interpr or args.learn_missing_interpr) and args.skip_sql_test
    print("Testing:")
    inference(checkpoint_path, test_dataset, res_path=file_name, user_message=user_message, skip_sql=skip_sql)

if __name__ == "__main__":
    args = parse_args()
    init_seed(args.seed)

    wandb.init(
        project="AmbigSQL",
        name=args.exp_name if args.exp_name else "ambigsql-default",
        config=vars(args)
    )

    # Load datasets at the beginning for train and validation
    if args.mode == "all" or args.mode == "validation" or args.mode == "validation,test":
        train_dataset, val_dataset, test_dataset = load_finetuning_datasets(args)
        full_validation_dataset = val_dataset
        
        if len(val_dataset) > 100:
            print("Sampling validation dataset to 100 examples to speed up training (full validation will be used after training)")
            wandb.run.summary["validation_sampled"] = True
            val_dataset = val_dataset.select(range(100))
            full_validation_dataset = val_dataset

    if args.mode == "all":
        train(args, train_dataset, val_dataset)
    
    if args.mode == "all" or args.mode == "validation" or args.mode == "validation,test":
        if args.skip_validation:
            # Use the last checkpoint without validation
            checkpoints = [f for f in os.listdir(args.output_dir) if f.startswith('checkpoint-')]
            checkpoints.sort(key=lambda x: int(x.split('-')[1]))
            last_checkpoint = os.path.join(args.output_dir, checkpoints[-1])
            top_k_checkpoints = [(0.0, last_checkpoint)]  # Dummy score with last checkpoint path
        else:
            # full_validation_dataset = val_dataset
            top_k_checkpoints = validation(args, full_validation_dataset)

    if args.mode == "all" or args.mode == "test" or args.mode == "validation,test":
        if args.test_checkpoint:
            checkpoint = args.test_checkpoint
        else:
            checkpoint = top_k_checkpoints[0][1]

        if args.dataset_type_test == 'all' or args.dataset_type_test == 'ambrosia':
            # Use test predictions for AMBROSIA
            args.dataset_type = "ambrosia"
            _, _, test_dataset = load_finetuning_datasets(args)
            print("AMBROSIA test")
            test(args, checkpoint, test_dataset, dataset_type="ambrosia")
            
        if args.dataset_type_test == 'all' or args.dataset_type_test == 'ambiqt':
            # Use test predictions for AmbiQT
            args.dataset_type = "ambiqt"
            _, _, test_dataset = load_finetuning_datasets(args)
            print("AmbiQT test")
            test(args, checkpoint, test_dataset, dataset_type="ambiqt")

    wandb.finish()