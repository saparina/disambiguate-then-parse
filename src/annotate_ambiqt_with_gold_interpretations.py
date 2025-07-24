import argparse
import json
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

from dataset import load_baseline_dataset
from utils import (
    init_model,
    get_generation_config,
    generate_from_prompt,
    generate_and_evaluate_sql
)
from prompts import user_message_interpretations_gen, user_message_sql

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate AmbiQT interpretations.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/ambiqt_gold_interpretations"))

    # Model initialization parameters
    parser.add_argument("--dtype", type=str, default="auto",
                       choices=["auto", "float16", "bfloat16", "float32"],
                       help="Data type for model")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--max_seq_length", type=int, default=8192,
                       help="Maximum sequence length")
    parser.add_argument("--chat_template", type=str, help="Override default chat template")

    parser.add_argument("--backend", type=str, choices=["unsloth", "tgi"], default="unsloth",
                       help="Backend to use for model inference")
    parser.add_argument("--tgi_url", type=str, default="http://localhost:8080/v1/",
                       help="URL for TGI API endpoint")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token")
    return parser.parse_args()

def generate_interpretations(model, tokenizer, example, generation_config, num_attempts=5):
    """Generate interpretations for each gold query in the example"""
    interpretations = []
    if example["nl_synonyms"] is None:
        return interpretations
    
    for idx in range(len(example['gold_queries'])):
        found = False
        for _ in range(num_attempts):
            # Generate interpretation
            messages = [
                {"role": "user", "content": user_message_interpretations_gen.format(
                    example["question"], example["nl_synonyms"][idx])}
            ]
            
            interpretation = generate_from_prompt(
                model, tokenizer, messages, generation_config
            ).strip()

            # Generate and evaluate SQL using utility function
            try:
                sql_result = generate_and_evaluate_sql(
                    model=model,
                    tokenizer=tokenizer,
                    db_dump=example["db_dump"],
                    text=interpretation,
                    db_file=example['db_file'],
                    gold_queries=[example['gold_queries'][idx]],
                    generation_config=generation_config,
                    verbose=False
                )
                
                if sql_result["success"] and sql_result.get("metrics", {}).get("f1_score", 0) == 1:
                    found = True
                    break
            except Exception as e:
                print(f"Error in SQL generation/evaluation: {str(e)}")
                continue
                
        interpretations.append((interpretation, found))

    return interpretations

def inference(args):
    # Initialize model and tokenizer using the common init_model function
    model, tokenizer = init_model(
        args,
        for_inference=True
    )

    # Load dataset
    dataset = load_baseline_dataset(
        dataset_type="ambiqt",
        for_train=True if args.split == "train" else False
    )

    if args.sample:
        random.shuffle(dataset)
        dataset = dataset.select(range(args.sample))

    # Get generation config
    generation_config = get_generation_config(args, model)

    # Setup output path
    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_name = args.output_dir / construct_output_filename(args)

    results = []

    for example in tqdm(dataset):
        interpretations = generate_interpretations(
            model, tokenizer, example, generation_config
        )
        
        if interpretations:
            result = {   
                "db_file": example['db_file'],
                "db_dump": example["db_dump"],
                "question": example["question"],
                "interpretations": interpretations,
                "gold_queries": example["gold_queries"],
                "is_ambiguous": example["is_ambiguous"]
            }
            results.append(result)
        
        if len(results) % 1000 == 0:
            save_results(file_name.with_name(f"{file_name.stem}_{len(results)}"), results)

    print(f"Generated {len(results)} interpretations")
    save_results(file_name, results)

def construct_output_filename(args):
    """Construct output filename based on arguments"""
    parts = [
        "interpretations",
        args.model_name.split('/')[-1].lower(),
        f"seed{args.seed}",
    ]
    
    if args.exp_name:
        parts.append(args.exp_name)
    if args.split:
        parts.append(args.split)
    if args.load_in_4bit:
        parts.append("4bit")
    if args.dtype != "auto":
        parts.append(args.dtype)
    if args.backend == "tgi":
        parts.append("tgi")
        
    return "_".join(parts) + ".json"

def save_results(file_path, results):
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    init_seed(args.seed)
    inference(args)