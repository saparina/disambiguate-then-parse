import argparse
import json
import os
import torch
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import numpy as np

from utils import (
    init_model, 
    get_generation_config,
    EvaluationMetricsTracker
)
from prompts import user_message_sql
from utils.metrics import count_unique_results, evaluate_predicted_statements, remove_duplicate_results
from utils.model_utils import generate_from_prompt
from utils.output_parsers import parse_statements_llama

def parse_args():
    parser = argparse.ArgumentParser(description="Process results and generate SQL queries")
    
    # Input/Output arguments
    parser.add_argument("--test_predictions", type=str, required=True,
                       help="Path to file with test results")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/final_interpretations_with_sql"),
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
    parser.add_argument("--use_existing_sql_prediction", action="store_true",
                       help="Use existing SQL predictions instead of generating new ones")
    
    # Backend settings
    parser.add_argument("--backend", type=str, choices=["unsloth", "tgi"], default="tgi",
                       help="Backend to use for model inference")
    parser.add_argument("--tgi_url", type=str, default="http://localhost:8080/v1/",
                       help="URL for TGI API endpoint")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token")
    
    args = parser.parse_args()
    return args

def process_test_results(
    model,
    tokenizer,
    test_predictions: str,
    generation_config: Dict,
    use_existing_sql_prediction: bool = False
) -> Dict:
    """Process test results file and generate/validate SQL for interpretations"""
    
    with open(test_predictions) as f:
        data = json.load(f)
    
    all_results = data.get("all_results", [])
    
    metrics_tracker = EvaluationMetricsTracker()
    
    statistics = {
        "total_examples": 0,
        "avg_interpretations": 0,
        "avg_unique_execution_results": 0,
        "avg_matched_gold_sql": 0
    }
    
    def process_single_interpretation(interp, example, model, tokenizer, generation_config):
        """Process a single interpretation"""
            
        messages = [{
            "role": "user", 
            "content": user_message_sql.format(example["db_dump"], interp)
        }]

        sql_prediction = generate_from_prompt(
            model, tokenizer, messages, generation_config
        )

        if sql_prediction is None:
            return {
                "success": False,
                "error": "Generation failed",
                "sql_queries": [],
                "metrics": None,
                "original_prediction": None
            }
        
        # Parse and clean SQL queries
        sql_queries = parse_statements_llama(sql_prediction)
        sql_queries = [q for q in sql_queries if q.lower().strip().startswith("select")]
        sql_queries = list(set(sql_queries))  # Remove duplicates

        result = {
            "sql_queries": sql_queries,
            "original_prediction": sql_prediction,
        }
        return result
    
    processed_results = []
    all_num_unique_results = []
    all_matched_gold_sql = []

    
    for example in tqdm(all_results):
        statistics["total_examples"] += 1
        
        result = {
            "db_file": example["db_file"],
            "db_dump": example["db_dump"],
            "question": example["question"],
            "gold_queries": example.get("gold_queries", []),
            "is_ambiguous": example.get("is_ambiguous", True),
            "ambig_type": example.get("ambig_type", "total"),
            "predicted_interpr": example["predicted_interpr"],
            "initial_generated_interpr": example.get("initial_generated_interpr", []),
            "generated_interpretations_sql": example.get("generated_interpretations_sql", {}),
            "all_interpretations": example["all_interpretations"],
            "all_interpretations_sql": example["all_interpretations_sql"],
            "predicted_interpr_sql": example["predicted_interpr_sql"]
        }
        
        
        matched_gold_sql = 0
      
        all_sql_queries = []
        if not use_existing_sql_prediction:
            # Generate new SQL predictions for initial interpretations
            if "initial_generated_interpr" in example:
                for interp in example["initial_generated_interpr"]:
                    if not "all possible interpretations are covered" in interp.lower().strip() and not "no interpretation" in interp.lower().strip():
                        sql_result = process_single_interpretation(
                            interp, example, model, tokenizer, generation_config
                        )
                        all_sql_queries.extend(sql_result["sql_queries"])
                        result["generated_interpretations_sql"][interp] = sql_result["sql_queries"]
                        result["all_interpretations_sql"][interp] = sql_result["sql_queries"]
        else:
            # Use existing SQL predictions
            for interpr, sql_query in result["generated_interpretations_sql"].items():
                all_sql_queries.extend(sql_query)
                result["all_interpretations_sql"][interpr] = sql_query
            result["generated_interpretations_sql"] = example["generated_interpretations_sql"]

        # Always process predicted interpretations
        for interp in example["predicted_interpr"]:
            if not "all possible interpretations are covered" in interp.lower().strip() and not "no interpretation" in interp.lower().strip():
                sql_result = process_single_interpretation(
                    interp, example, model, tokenizer, generation_config
                )
                all_sql_queries.extend(sql_result["sql_queries"])
                result["predicted_interpr_sql"][interp] = sql_result["sql_queries"]
                result["all_interpretations_sql"][interp] = sql_result["sql_queries"]
        
        
        try:
            # Evaluate all SQL queries together
            sql_metrics = evaluate_predicted_statements(
                example["db_file"],
                all_sql_queries,
                example.get("gold_queries", []),
                remove_duplicates_predictions=False,
                verbose=False,
                return_pred_exec_outputs=True
            )
            
            pred_exec_outputs = remove_duplicate_results(sql_metrics["pred_exec_outputs"])
            num_unique_results = count_unique_results(list(pred_exec_outputs.values()))
            
            result["num_unique_execution_results"] = num_unique_results
            sql_metrics_cleaned = {k: v for k, v in sql_metrics.items() 
                                 if k not in ["pred_exec_outputs", "gold_exec_outputs"]}
            result["interpretation_level_metrics"] = sql_metrics_cleaned
            
            all_num_unique_results.append(num_unique_results)
            all_matched_gold_sql.append(matched_gold_sql)
            
            main_key = "ambig" if example.get("is_ambiguous", True) else "unambig"
            ambig_type = example.get("ambig_type", "total")
            
            metrics_tracker.update_metrics(main_key, ambig_type, sql_metrics)
            
        except Exception as e:
            print(f"Error computing metrics for question: {example['question']}")
            print(f"Error: {str(e)}")
            
            main_key = "ambig" if example.get("is_ambiguous", True) else "unambig"
            ambig_type = example.get("ambig_type", "total")
            
            metrics_tracker.add_zero_metrics(main_key, ambig_type)
            result["num_unique_execution_results"] = 0
            result["interpretation_level_metrics"] = {}
            
            all_num_unique_results.append(0)
            all_matched_gold_sql.append(0)
        
        result["matched_gold_sql"] = matched_gold_sql
        processed_results.append(result)
    
    # Calculate statistics
    statistics["avg_interpretations"] = np.mean([len(res["all_interpretations"]) for res in processed_results])
    statistics["avg_unique_execution_results"] = np.mean(all_num_unique_results)
    statistics["avg_matched_gold_sql"] = np.mean(all_matched_gold_sql)
    
    aggregated_metrics = metrics_tracker.get_aggregated_metrics()
    metrics_tracker.print_summary()
    
    return {
        "results": processed_results,
        "metrics": aggregated_metrics,
        "statistics": statistics
    }

def construct_output_filename(args, test_predictions: str) -> str:
    """Construct output filename"""
    base_name = Path(test_predictions).stem
    sql_model_name = args.model_name.split('/')[-1].lower()
    
    parts = [
        "processed",
        base_name,
        f"sql_{sql_model_name}",
    ]
    
    if args.exp_name:
        parts.append(args.exp_name)
    if args.load_in_4bit:
        parts.append("4bit")
    if args.dtype != "auto":
        parts.append(args.dtype)
    if args.backend == "tgi":
        parts.append("tgi")
        
    return "_".join(parts) + ".json"

def main():
    args = parse_args()
    torch.manual_seed(42)  # Fixed seed for reproducibility
    
    # Initialize model and get generation config
    model, tokenizer = init_model(args, for_inference=True)
    generation_config = get_generation_config(args, model)
    
    # Process test results
    results = process_test_results(
        model=model,
        tokenizer=tokenizer,
        test_predictions=args.test_predictions,
        generation_config=generation_config,
        use_existing_sql_prediction=args.use_existing_sql_prediction
    )
    
    # Save results
    output_file = construct_output_filename(args, args.test_predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / output_file
    
    # Add args to results
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    results["args"] = args_dict
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    stats = results["statistics"]
    print("\nProcessing Summary:")
    print(f"Total examples processed: {stats['total_examples']}")
    print(f"Average interpretations per example: {stats['avg_interpretations']:.2f}")
    print(f"Average unique execution results per example: {stats['avg_unique_execution_results']:.2f}")
    print(f"Average matched gold SQL per example: {stats['avg_matched_gold_sql']:.2f}")
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main() 