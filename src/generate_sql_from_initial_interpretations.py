import argparse
import json
import torch
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import (
    init_model, 
    get_generation_config,
    EvaluationMetricsTracker,
    generate_and_evaluate_sql
)
from prompts import user_message_sql
from utils.metrics import count_unique_results, evaluate_predicted_statements, remove_duplicate_results

def parse_args():
    parser = argparse.ArgumentParser(description="Generate SQL queries from initial interpretations")
    
    # Input/Output arguments
    parser.add_argument("--interpretation_file", type=str, required=True,
                       help="Path to file with generated interpretations")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/initial_interpretations_with_sql"),
                       help="Directory to save results")
    parser.add_argument("--exp_name", type=str,
                       help="Experiment name for results file")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                       help="Model to use for SQL generation")
    parser.add_argument("--dtype", type=str, default="auto",
                       choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--chat_template", type=str,
                       help="Override default chat template")
    
    # Processing options
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing SQL queries")
    
    # Other settings
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    parser.add_argument("--backend", type=str, choices=["unsloth", "tgi"], default="unsloth",
                       help="Backend to use for model inference")
    parser.add_argument("--tgi_url", type=str, default="http://localhost:8080/v1/",
                       help="URL for TGI API endpoint")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token")

    # New argument for load_from
    parser.add_argument("--load_from", type=str, help="Path to existing results file to continue from")
    
    # Add num_workers argument
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of worker threads for parallel processing")
    
    args = parser.parse_args()
    return args

def load_previous_results(output_path: Path, metrics_tracker: EvaluationMetricsTracker) -> tuple[list, list, list, EvaluationMetricsTracker]:
    """Load results from a previous run and recalculate metrics"""
    if not output_path.exists():
        print(f"No previous results found at {output_path}")
        return [], [], [], metrics_tracker
    
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
    
    return results, all_num_unique_results, all_matched_gold_sql, metrics_tracker

def process_single_interpretation(interp, example, model, tokenizer, generation_config):
    """Process a single interpretation"""
    return generate_and_evaluate_sql(
        model=model,
        tokenizer=tokenizer,
        db_dump=example["db_dump"],
        text=interp["interpretation"],
        db_file=example["db_file"],
        gold_queries=example.get("gold_queries"),
        generation_config=generation_config,
        prompt_template=user_message_sql
    )

def process_single_example(example, model, tokenizer, generation_config, num_workers=4):
    """Process a single example and its interpretations in parallel"""
    result = {
        "db_file": example["db_file"],
        "db_dump": example["db_dump"],
        "question": example["question"],
        "gold_queries": example.get("gold_queries", []),
        "is_ambiguous": example.get("is_ambiguous", True),
        "ambig_type": example.get("ambig_type", "total"),
        "interpretations": example["interpretations"]
    }
    
    # Process interpretations in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_single_interpretation, interp, example, model, tokenizer, generation_config)
            for interp in result["interpretations"]
        ]
        
        matched_gold_sql = 0
        for interp_idx, future in enumerate(as_completed(futures)):
            try:
                sql_result = future.result()
                result["interpretations"][interp_idx]["metrics"] = sql_result["metrics"]
                result["interpretations"][interp_idx]["sql_queries"] = sql_result["sql_queries"]
                if sql_result["metrics"]["recall"] > 0:
                    matched_gold_sql += 1
            except Exception as e:
                print(f"Error processing interpretation {interp_idx}: {str(e)}")
                import pdb; pdb.set_trace()
    
    all_sql_queries = []
    for interp_result in example["interpretations"]:
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
        num_unique_results = count_unique_results(list(pred_exec_outputs.values()))

        result["num_unique_execution_results"] = num_unique_results
        sql_metrics_cleaned = {k: v for k, v in sql_metrics.items() if k not in ["pred_exec_outputs", "gold_exec_outputs"]}
        result["interpretation_level_metrics"] = sql_metrics_cleaned
        
        # Get ambiguity type from the example
        main_key = "ambig" if example.get("is_ambiguous", True) else "unambig"
        ambig_type = example.get("ambig_type", "total")
        
        return result, num_unique_results, matched_gold_sql, main_key, ambig_type, sql_metrics
    except Exception as e:
        print(f"Error computing metrics for question: {example['question']}")
        print(f"Error: {str(e)}")
        
        # Get ambiguity type from the example
        main_key = "ambig" if example.get("is_ambiguous", True) else "unambig"
        ambig_type = example.get("ambig_type", "total")
        
        result["num_unique_execution_results"] = 0
        result["interpretation_level_metrics"] = {}
        
        return result, 0, 0, main_key, ambig_type, {}

def process_interpretation_file(
    model,
    tokenizer,
    interpretation_file: str,
    generation_config: Dict,
    output_path: Path = None,
    load_from: Path = None,
    num_workers: int = 4
) -> Dict:
    """Process file with interpretations and generate/validate SQL"""
    
    metrics_tracker = EvaluationMetricsTracker()
    
    # Load previous results if specified
    load_path = load_from if load_from else (output_path if output_path else None)
    processed_results, all_num_unique_results, all_matched_gold_sql, metrics_tracker = (
        load_previous_results(load_path, metrics_tracker) if load_path else ([], [], [], metrics_tracker)
    )
    
    # Create a set of processed examples
    processed_examples = {(r['db_file'], r['question']) for r in processed_results}
    print(f"Loaded {len(processed_results)} previously processed examples")
    
    # Initialize statistics
    statistics = {
        "total_examples": len(processed_results),
        "avg_interpretations": 0,
        "avg_unique_execution_results": 0,
        "avg_matched_gold_sql": 0
    }
    
    # Load all examples at once
    with open(interpretation_file) as f:
        data = json.load(f)
        all_examples = data["results"]
    
    # Filter out already processed examples
    examples_to_process = [
        example for example in all_examples 
        if (example['db_file'], example['question']) not in processed_examples
    ]
    
    if not examples_to_process:
        print("No new examples to process")
        return {
            "results": processed_results,
            "metrics": metrics_tracker.get_aggregated_metrics(),
            "statistics": statistics
        }
    
    print(f"Processing {len(examples_to_process)} examples")
    
    for example_idx, example in enumerate(tqdm(examples_to_process)):

        result, num_unique_results, matched_gold_sql, main_key, ambig_type, sql_metrics = process_single_example(
            example, model, tokenizer, generation_config, num_workers
        )
        
        result["matched_gold_sql"] = matched_gold_sql
        
        processed_results.append(result)
            
        # Save intermediate results every 100 examples
        if output_path and example_idx > 0 and example_idx % 500 == 0:
            current_statistics = {
                "total_examples": len(processed_results),
                "avg_interpretations": np.mean([len(r["interpretations"]) for r in processed_results]),
                "avg_unique_execution_results": np.mean(all_num_unique_results),
                "avg_matched_gold_sql": np.mean(all_matched_gold_sql)
            }
            
            intermediate_results = {
                "results": processed_results,
                "metrics": metrics_tracker.get_aggregated_metrics(),
                "statistics": current_statistics
            }
            
            # Save to intermediate file
            intermediate_path = output_path.with_stem(f"{output_path.stem}_{len(processed_results)}")
            with open(intermediate_path, 'w', encoding='utf-8') as f:
                json.dump(intermediate_results, f, indent=2, ensure_ascii=False)

    # Calculate final averages
    statistics["avg_interpretations"] = np.mean([len(interp["interpretations"]) for interp in processed_results])
    statistics["avg_unique_execution_results"] = np.mean(all_num_unique_results)
    statistics["avg_matched_gold_sql"] = np.mean(all_matched_gold_sql)
    
    aggregated_metrics = metrics_tracker.get_aggregated_metrics()
    metrics_tracker.print_summary()

    return {
        "results": processed_results,
        "metrics": aggregated_metrics,
        "statistics": statistics
    }

def extract_params_from_filename(interpretation_file: str) -> dict:
    """Extract parameters from interpretations filename"""
    basename = Path(interpretation_file).stem
    parts = basename.split('_')
    
    params = {
        "dataset_type": None,
        "split": None,
        "no_database": False,
        "filter_gold": False,
        "filter_interpr": False,
        "load_in_4bit": False,
        "dtype": "auto",
        "backend": "unsloth",
        "model_name": None,
        "ambrosia_file": None
    }
    
    # Extract dataset type and split (usually in format: interpretations_modelname_seed42_datasettype_split...)
    try:
        params["dataset_type"] = parts[3]  # After "interpretations_modelname_seed42"
        params["split"] = parts[4]
    except IndexError:
        print("Warning: Could not extract dataset_type or split from filename")
    
    # Check for other parameters in filename
    if "no_database" in parts:
        params["no_database"] = True
    if "filtered_gold" in parts:
        params["filter_gold"] = True
    if "filtered_interpr" in parts:
        params["filter_interpr"] = True
    if "4bit" in parts:
        params["load_in_4bit"] = True
    if "float16" in parts:
        params["dtype"] = "float16"
    elif "bfloat16" in parts:
        params["dtype"] = "bfloat16"
    if "tgi" in parts:
        params["backend"] = "tgi"
    
    # Load the file to get additional parameters
    try:
        with open(interpretation_file, 'r') as f:
            data = json.load(f)
            args = data.get("args", {})
            params["model_name"] = args.get("model_name")
            params["ambrosia_file"] = args.get("ambrosia_file")
            
            # Handle ambrosia_resplit case
            if params["dataset_type"] == "ambrosia" and params["ambrosia_file"] and "resplit" in params["ambrosia_file"]:
                params["dataset_type"] = "ambrosia_resplit"
    except Exception as e:
        print(f"Warning: Could not load file to extract additional parameters: {e}")
        
    return params

def construct_output_filename_sql(args, interpretations_model_name: str) -> str:
    """Construct output filename"""
    # Extract model names after last '/'
    sql_model_name = args.model_name.split('/')[-1].lower()
    interp_model_name = interpretations_model_name.split('/')[-1].lower()
    
    parts = [
        "sql",
        f"interp_{interp_model_name}",
        f"sql_{sql_model_name}",
        f"seed{args.seed}",
        args.dataset_type,
        args.split
    ]
    
    if args.exp_name:
        parts.append(args.exp_name)
    if args.no_database:
        parts.append("no_database")
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

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract model name and dataset type from input filename
    input_path = Path(args.interpretation_file)
    model_name = input_path.parent.name  # Get model name from parent directory
    
    # Extract parameters from input filename
    params = extract_params_from_filename(args.interpretation_file)
    dataset_type = params["dataset_type"]
    
    # Create model and dataset specific output directory
    model_output_dir = output_dir / model_name / dataset_type
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct output filename
    output_filename = construct_output_filename_sql(args, params.get("model_name", "unknown"))
    output_path = model_output_dir / output_filename
    
    # Initialize model and get generation config
    model, tokenizer = init_model(args, for_inference=True)
    generation_config = get_generation_config(args, model)
    
    
    # Update args with extracted parameters
    for key, value in params.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    
    # Extract the original model name from the results
    with open(args.interpretation_file) as f:
        interp_data = json.load(f)
    interpretations_model_name = interp_data.get("args", {}).get("model_name", "unknown_model")
    
    # Process interpretations with parallel processing
    results = process_interpretation_file(
        model=model,
        tokenizer=tokenizer,
        interpretation_file=args.interpretation_file,
        generation_config=generation_config,
        output_path=output_path,
        load_from=Path(args.load_from) if args.load_from else None,
        num_workers=args.num_workers
    )
    
    # Add args to results
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    results["args"] = args_dict
    
    # Save final results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Clean up intermediate files
    intermediate_pattern = f"{output_path.stem}_*{output_path.suffix}"
    for intermediate_file in output_path.parent.glob(intermediate_pattern):
        try:
            intermediate_file.unlink()
            print(f"Cleaned up intermediate file: {intermediate_file}")
        except Exception as e:
            print(f"Failed to delete {intermediate_file}: {e}")
    
    # Print statistics
    stats = results["statistics"]
    print("\nProcessing Summary:")
    print(f"Average interpretations per example: {stats['avg_interpretations']:.2f}")
    print(f"Average unique execution results per example: {stats['avg_unique_execution_results']:.2f}")
    print(f"Average matched gold SQL per example: {stats['avg_matched_gold_sql']:.2f}")
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main() 