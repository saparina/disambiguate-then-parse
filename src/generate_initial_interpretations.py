import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np

from datasets import Dataset
from tqdm import tqdm
import torch

from prompts import user_message_interpr_ambig, user_message_sql
from dataset import load_interpretations_dataset

from utils import (
    init_model,
    get_generation_config,
    generate_from_prompt,
    EvaluationMetricsTracker,
    generate_and_evaluate_sql,
    parse_interpretations,
    count_unique_results,
    evaluate_predicted_statements, 
    remove_duplicate_results
)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate initial interpretations")
    
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--dataset_type", type=str, choices=["ambrosia", "ambiqt"], required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"], 
                       help="'train' for train for ambrosia and train+validation for ambiqt, 'validation' for validation for ambrosia, 'test' for test split")
    
    # Output directory is now fixed
    parser.add_argument("--output_dir", type=str, default="outputs/initial_interpretations",
                       help="Directory to save results")
    
    # Dataset related arguments
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--ambrosia_file", type=str, default="data/ambrosia/data/ambrosia_resplit.csv")
    
    # Filtering options
    parser.add_argument("--filter_gold", action="store_true", help="Filter examples with different execution results")
    parser.add_argument("--filter_interpr", action="store_true", help="Only process questions with gold interpretations")
    
    # Model configuration
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "auto"], default="auto")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--chat_template", type=str, help="Override default chat template")
    
    # Other options
    parser.add_argument("--exp_name", type=str, help="Custom name for the experiment")
    parser.add_argument("--validate_sql", action="store_true", help="Generate and validate SQL queries")
    parser.add_argument("--seed", type=int, default=42)

    # Generation parameters - all optional, will use model's defaults if not specified
    parser.add_argument("--max_new_tokens", type=int, help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", type=bool, help="Whether to use sampling for generation")
    parser.add_argument("--num_beams", type=int, help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, help="Top-k sampling parameter")
    
    parser.add_argument("--backend", type=str, choices=["unsloth", "tgi"], default="unsloth",
                       help="Backend to use for model inference")
    parser.add_argument("--tgi_url", type=str, default="http://localhost:8080/v1/",
                       help="URL for TGI API endpoint")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token")
    
    parser.add_argument("--load_from", type=str, help="Path to existing results file to continue from")
    
    return parser.parse_args()


def generate_interpretations(model, tokenizer, example: Dict, args) -> Dict:
    """Generate interpretations for a single example"""
    messages = [{"role": "user", "content": user_message_interpr_ambig.format(example["db_dump"], example["question"])}]
    
    generation_config = get_generation_config(args, model)
    predictions = generate_from_prompt(
        model, 
        tokenizer, 
        messages, 
        generation_config,
        max_length=model.get_max_length()
    )
    
    if predictions is None:
        print(f"Warning: Failed to generate interpretation for database {example['db_file']} and question: {example['question']}")
        return {"results": [], "original_prediction": "", "execution_results": []}
    
    interpretations = parse_interpretations(predictions)
    interpretations = list(set(interpretations))
    
    results = []
    matched_gold_sql = 0
    if args.validate_sql:
        for interpretation in interpretations:
            sql_result = generate_and_evaluate_sql(
                model=model,
                tokenizer=tokenizer,
                db_dump=example["db_dump"],
                text=interpretation,
                db_file=example["db_file"],
                gold_queries=example.get("gold_queries"),
                generation_config=generation_config,
                prompt_template=user_message_sql
            )
            
            if sql_result["success"]:
                result = {
                    "interpretation": interpretation,
                    "sql_queries": sql_result["sql_queries"],
                    "metrics": sql_result["metrics"]
                }
                results.append(result)
                if sql_result["metrics"]["recall"] > 0:
                    matched_gold_sql += 1
        
    return {
        "results": results if args.validate_sql else [{"interpretation": i} for i in interpretations],
        "original_prediction": predictions,
        "num_interpretations": len(interpretations),  # Count interpretations per question
        "matched_gold_sql": matched_gold_sql
    }

def construct_output_filename_interpretations(args) -> str:
    """Construct output filename based on arguments"""
    # Extract model name after last '/'
    model_name = args.model_name.split('/')[-1].lower()
    
    parts = [
        f"initial_interpretations",
        model_name,
        f"seed{args.seed}",
        args.dataset_type,
        args.split
    ]
    
    if args.exp_name:
        parts.append(args.exp_name)
    if args.filter_gold:
        parts.append("filtered_gold")
    if args.filter_interpr:
        parts.append("filtered_interpr")
    if args.load_in_4bit:
        parts.append("4bit")
    if args.dtype != "auto":
        parts.append(args.dtype)
    if args.backend == "tgi":
        parts.append("tgi")
        
    return "_".join(parts) + ".json"

def save_results(results, metrics, statistics, args, output_path):
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    
    output = {
        "metrics": metrics,
        "statistics": statistics,
        "results": results,
        "args": args_dict
    }
    
    model_name = args.model_name.split('/')[-1].lower()
    dataset_type = args.dataset_type
    if dataset_type == "ambrosia" and "resplit" in args.ambrosia_file:
        dataset_type = "ambrosia_resplit"
    output_dir = output_path.parent / model_name / dataset_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / output_path.name, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def load_previous_results(output_path: Path, metrics_tracker: EvaluationMetricsTracker) -> tuple[list, list, EvaluationMetricsTracker]:
    """Load results from a previous run and recalculate metrics"""
    if not output_path.exists():
        print(f"No previous results found at {output_path}")
        return [], [], metrics_tracker
    
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    # Reconstruct statistics lists from actual results
    all_num_unique_results = []
    all_matched_gold_sql = []
    
    # Reset metrics tracker
    metrics_tracker = EvaluationMetricsTracker()
    
    print("Recalculating metrics from previous results...")
    for result in results:
        # Get num_unique_results and matched_gold_sql from actual results
        all_num_unique_results.append(result.get('num_unique_execution_results', 0))
        all_matched_gold_sql.append(result.get('matched_gold_sql', 0))
        
        # Update metrics tracker with interpretation level metrics
        main_key = "ambig" if result.get("is_ambiguous", True) else "unambig"
        ambig_type = result.get("ambig_type", "total")
        
        if result.get("interpretation_level_metrics"):
            metrics_tracker.update_metrics(
                main_key,
                ambig_type,
                result["interpretation_level_metrics"]
            )
        else:
            metrics_tracker.add_zero_metrics(main_key, ambig_type)
    
    print(f"Loaded and recalculated metrics for {len(results)} previous results")
    
    return results, all_num_unique_results, metrics_tracker


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    # Initialize model and load dataset
    model, tokenizer = init_model(args, for_inference=True)
    dataset = load_interpretations_dataset(args)
    
    # Generate output file name and create directory
    output_file = construct_output_filename_interpretations(args)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = Path(args.output_dir) / output_file

    # Initialize metrics tracker if we're validating SQL
    metrics_tracker = EvaluationMetricsTracker() if args.validate_sql else None
    
    # Load previous results if specified or try current output path
    load_path = Path(args.load_from) if args.load_from else output_path
    results, num_unique_results, metrics_tracker = load_previous_results(load_path, metrics_tracker)
    
    # Create a set of (db_file, question) tuples for processed examples
    processed_examples = {(r['db_file'], r['question']) for r in results}
    
    for example in tqdm(dataset):
        # Skip if we already processed this db_file + question combination
        if (example['db_file'], example['question']) in processed_examples:
            continue
            
        interpretation = generate_interpretations(model, tokenizer, example, args)
        
        if interpretation:
            result = {
                "db_file": example.get('db_file'),
                "db_dump": example.get("db_dump"),
                "question": example["question"],
                "interpretations": interpretation["results"],
                "gold_queries": example.get("gold_queries"),
                "is_ambiguous": example.get("is_ambiguous", True),
                "ambig_type": example.get("ambig_type", "total"),
                "num_interpretations": interpretation.get("num_interpretations", 0),  # Get number of interpretations
                "matched_gold_sql": interpretation.get("matched_gold_sql", 0),  # Get number of matched gold SQL,
                "original_prediction": interpretation["original_prediction"]
            }
            results.append(result)
            
            # Update metrics if SQL validation was performed
            if args.validate_sql and interpretation["results"]:
                main_key = "ambig" if example.get("is_ambiguous", True) else "unambig"
                ambig_type = example.get("ambig_type", "total")
                
                # Get all SQL queries from all interpretations
                all_sql_queries = []
                for interp_result in interpretation["results"]:
                    if "sql_queries" in interp_result:
                        all_sql_queries.extend(interp_result["sql_queries"])
                
                try:
                    sql_metrics = evaluate_predicted_statements(
                        example["db_file"],
                        all_sql_queries,
                        example.get("gold_queries", []),
                        remove_duplicates_predictions=False,
                        verbose=False,
                        return_pred_exec_outputs=True
                    )
                    
                    # Count matched gold SQL
                    pred_exec_outputs = remove_duplicate_results(sql_metrics["pred_exec_outputs"])
                    num_unique_results.append(count_unique_results(list(pred_exec_outputs.values())))

                    result["num_unique_execution_results"] = num_unique_results[-1]
                    sql_metrics_cleaned = {k: v for k, v in sql_metrics.items() if k not in ["pred_exec_outputs", "gold_exec_outputs"]}
                    result["interpretation_level_metrics"] = sql_metrics_cleaned
                    
                    # Update metrics using the tracker
                    metrics_tracker.update_metrics(main_key, ambig_type, sql_metrics)
                except Exception as e:
                    print(f"Error computing metrics for question: {example['question']}")
                    print(f"Error: {str(e)}")
                    metrics_tracker.add_zero_metrics(main_key, ambig_type)

                    result["num_unique_execution_results"] = 0
                    result["interpretation_level_metrics"] = {}
        
        # Save intermediate results periodically
        if len(results) % 100 == 0:
            current_statistics = {
                    "total_examples": len(results),
                    "avg_interpretations": np.mean([r["num_interpretations"] for r in results]),
                    "avg_unique_execution_results": np.mean(num_unique_results),
                    "avg_matched_gold_sql": np.mean([r["matched_gold_sql"] for r in results])
            }
            save_results(
                results, 
                metrics_tracker.get_aggregated_metrics() if metrics_tracker else {}, 
                current_statistics,
                args, 
                output_path.with_stem(f"{output_path.stem}_{len(results)}")
            )
        
        if not interpretation:
            print(f"Warning: Failed to generate interpretation for question: {example['question']}")
    
    # Print final metrics summary if available
    if metrics_tracker:
        metrics_tracker.print_summary()
    
    # Print summary of interpretations
    statistics = {
        "total_examples": len(results),
        "avg_interpretations": np.mean([r["num_interpretations"] for r in results]),
        "avg_unique_execution_results": np.mean(num_unique_results),
        "avg_matched_gold_sql": np.mean([r["matched_gold_sql"] for r in results])
    }

    print(f"Average interpretations per example: {statistics['avg_interpretations']}")
    print(f"Average unique execution results per example: {statistics['avg_unique_execution_results']}")
    print(f"Average matched gold SQL per example: {statistics['avg_matched_gold_sql']}")

    # Save final results with aggregated metrics (empty if no SQL validation)
    save_results(
        results, 
        metrics_tracker.get_aggregated_metrics() if metrics_tracker else {}, 
        statistics,
        args, 
        output_path
    )
    
    # Clean up intermediate files
    intermediate_pattern = f"{output_path.stem}_*{output_path.suffix}"
    for intermediate_file in output_path.parent.glob(intermediate_pattern):
        try:
            intermediate_file.unlink()
            print(f"Cleaned up intermediate file: {intermediate_file}")
        except Exception as e:
            print(f"Failed to delete {intermediate_file}: {e}")
    
    print(f"Generated interpretations for {len(results)} examples")

if __name__ == "__main__":
    main()