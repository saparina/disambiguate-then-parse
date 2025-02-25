import json
import argparse
import random
from tqdm import tqdm
from typing import Dict
from pathlib import Path

from utils import (
    init_model,
    get_generation_config,
    EvaluationMetricsTracker,
    generate_and_evaluate_sql
)
from dataset import load_baseline_dataset, filter_db_dump

from prompts import (
    user_message_sql_ambig,
    user_message_sql_ambig_icl,
    user_message_sql
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate end-to-end text-to-SQL parsers in zero-shot and few-shot settings")
    # Model and output arguments
    parser.add_argument("--model_name", type=str, required=True,
                       help="Model to evaluate")
    parser.add_argument("--res_dir", type=Path, default=Path("outputs"),
                       help="Directory for results")
    parser.add_argument("--exp_name", type=str,
                       help="Experiment name for results file")

    # Dataset arguments
    parser.add_argument("--dataset_type", type=str, choices=["ambrosia", "ambiqt", "all"],
                       help="Dataset type to evaluate on")
    parser.add_argument("--filter_gold", action="store_true",
                       help="Whether to filter gold queries (for AmbiQT)")
    
    # Evaluation settings
    parser.add_argument("--icl_examples", type=int, default=0,
                       help="Number of in-context learning examples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Model initialization parameters
    parser.add_argument("--dtype", type=str, default="auto",
                       choices=["auto", "float16", "bfloat16", "float32"],
                       help="Data type for model")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--max_seq_length", type=int, default=8192,
                       help="Maximum sequence length")
    parser.add_argument("--chat_template", type=str, help="Override default chat template")
    
    # Add task type explicitly
    parser.add_argument("--task_type", type=str, default="text2sql_baseline",
                       help="Task type for dataset loading")
    
    # Replace prompt_ambiguity_mentioned with prompt_type
    parser.add_argument("--prompt_type", type=str, 
                       choices=["ambiguity_mentioned", "std_text2sql"],
                       default="ambiguity_mentioned",
                       help="Type of prompt to use")
    
    # Add ambrosia filtering option
    parser.add_argument("--ambrosia_file", type=str, default="data/ambrosia/data/ambrosia.csv",
                       help="Path to Ambrosia dataset file")
    
    parser.add_argument("--ambrosia_question_type", type=str,
                       choices=["ambig", "unambig"],
                       help="Filter Ambrosia dataset to specific question type")
    
    # Add verbose option
    parser.add_argument("--verbose", action="store_true",
                       help="Print verbose output")
    
    parser.add_argument("--backend", type=str, choices=["unsloth", "tgi"], default="unsloth",
                       help="Backend to use for model inference")
    parser.add_argument("--tgi_url", type=str, default="http://localhost:8080/v1/",
                       help="URL for TGI API endpoint")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token")
    
    args = parser.parse_args()
    
    
    return args

def sample_icl_examples(train_dataset, num_examples: int = 3) -> str:
    """Sample ICL examples from training data"""
    train_df = train_dataset.to_pandas()
    
    # Group by ambiguous question
    grouped = train_df.groupby('ambig_question')
    unique_questions = list(grouped.groups.keys())
    
    # Sample questions
    sampled_questions = random.sample(unique_questions, num_examples)
    
    prompt = ""
    for i, question in enumerate(sampled_questions, 1):
        group = grouped.get_group(question)
        ambig_row = group[group['is_ambiguous'] == True].iloc[0]
        interpretations = group[group['is_ambiguous'] == False]
        
        # Format example
        db_dump = filter_db_dump(ambig_row['db_dump'], '\n'.join(ambig_row['gold_queries']))
        prompt += f"Example {i}:\nGiven the following SQLite database schema:\n\n{db_dump}"
        prompt += f"Answer the following:\n{ambig_row['question']}\n\nSQL query(s):\n"
        
        for query in ambig_row['gold_queries']:
            prompt += f"{query}\n\n"
            
        # Add unambiguous interpretations
        for _, row in interpretations.iterrows():
            ex = row.to_dict()
            prompt += f"Answer the following:\n{ex['question']}\n\nSQL query(s):\n"
            prompt += f"{ex['gold_queries'][0]}\n\n"
            
    return prompt.rstrip()

def evaluate_dataset(
    model,
    tokenizer,
    dataset,
    generation_config,
    icl_prompt: str = None,
    use_original_db: bool = False,
    prompt_type: str = "ambiguity_mentioned",
    verbose: bool = False
) -> Dict:
    """Evaluate model on dataset"""
    results = []
    metrics_tracker = EvaluationMetricsTracker()

    # Prepare base prompt based on type
    if prompt_type == "ambiguity_mentioned":
        prompt_template = user_message_sql_ambig_icl.replace("EXAMPLES", icl_prompt) if icl_prompt else user_message_sql_ambig
    else:
        prompt_template = user_message_sql
    
    for example in tqdm(dataset):
        # Prepare input
        db_dump = example["db_dump_original" if use_original_db else "db_dump"]
        
        sql_result = generate_and_evaluate_sql(
            model=model,
            tokenizer=tokenizer,
            db_dump=db_dump,
            text=example["question"],
            db_file=example["db_file"],
            gold_queries=example["gold_queries"],
            generation_config=generation_config,
            prompt_template=prompt_template,
            verbose=verbose
        )
        
        # Prepare result entry
        result = {
            "db_file": example['db_file'],
            "db_dump": db_dump,
            "question": example["question"],
            "prediction": sql_result["sql_queries"],
            "gold_queries": example['gold_queries'],
            "is_ambiguous": example["is_ambiguous"],
            "original_prediction": sql_result["original_prediction"]
        }
        
        # Get metrics and ambig_type
        main_key = "ambig" if example["is_ambiguous"] else "unambig"
        ambig_type = example.get("ambig_type", "vague")
        
        # Add ambig_type to result
        result["ambig_type"] = ambig_type
        
        if sql_result["success"]:
            if sql_result["metrics"]:
                metrics_tracker.update_metrics(main_key, ambig_type, sql_result["metrics"])
                result.update(sql_result["metrics"])
        else:
            # Handle failed predictions
            metrics_tracker.add_zero_metrics(main_key, ambig_type)
            result.update({
                "precision": 0.0,
                "recall": 0.0,
                "all_found": 0.0,
                "one_found": 0.0
            })
        
        results.append(result)
    
    # Get aggregated metrics
    aggregated_metrics = metrics_tracker.get_aggregated_metrics()
    
    return {"metrics": aggregated_metrics, "results": results}

def main():
    args = parse_args()
    args.res_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model and tokenizer
    model, tokenizer = init_model(args, for_inference=True)
    
    # Get generation config using existing function
    generation_config = get_generation_config(args, model)
    
    dataset_types = ["ambrosia", "ambiqt"] if args.dataset_type == "all" else [args.dataset_type]

    # Load ICL examples if needed
    icl_prompt = None
    if args.icl_examples > 0:
        train_dataset = load_baseline_dataset(args.dataset_type, for_train=True, ambrosia_file=args.ambrosia_file)
        icl_prompt = sample_icl_examples(train_dataset, num_examples=args.icl_examples)
    
    # Evaluate on specified dataset(s)
    results = {}
    
    for dataset_type in dataset_types:
        # Load appropriate test set based on dataset type
        test_dataset = load_baseline_dataset(
            dataset_type, 
            for_train=False,
            ambrosia_question_type=args.ambrosia_question_type,
            ambrosia_file=args.ambrosia_file
        )
        
        final_results = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset=test_dataset,
            generation_config=generation_config,
            icl_prompt=icl_prompt,
            prompt_type=args.prompt_type,
            verbose=args.verbose
        )
        
        # Update save path based on prompt type
        prompt_type_path = "ambiguity_prompt" if args.prompt_type == "ambiguity_mentioned" else "std_prompt"
        save_dir = args.res_dir / "baselines" / prompt_type_path
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract model name after last '/'
        model_name = args.model_name.split('/')[-1]
        
        # Create filename with model name and parameters
        filename = f"results_{model_name}_seed{args.seed}"
        if args.exp_name:
            filename += f"_exp{args.exp_name}"
        filename += f"_{dataset_type}"
        if args.icl_examples:
            filename += f"_icl{args.icl_examples}"
        if args.load_in_4bit:
            filename += "_4bit"
        if args.dtype != "auto":
            filename += f"_{args.dtype}"
        if args.backend == "tgi":
            filename += "_tgi"
        filename += ".json"
        
        # Convert args to dict and ensure all Path objects are strings
        args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        final_results["args"] = args_dict
        
        # Save results for current dataset
        with open(save_dir / filename, 'w') as f:
            json.dump(final_results, f, indent=4)
    
        # Print summary
        print(f"\nResults for {dataset_type}:")
        for main_key in final_results["metrics"]:
            print(f"\n{main_key.capitalize()}:")
            for ambig_type, metrics in final_results["metrics"][main_key].items():
                print(f"  {ambig_type.capitalize()}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value:.4f}")

if __name__ == "__main__":
    main() 