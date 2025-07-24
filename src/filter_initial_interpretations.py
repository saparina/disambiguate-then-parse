import argparse
import json
import random
from pathlib import Path
from typing import List, Dict
import sqlite3
from tqdm import tqdm
import numpy as np

from utils import (
    compare_query_results,  
    evaluate_predicted_statements, 
    remove_duplicate_results, 
    count_unique_results, 
    EvaluationMetricsTracker
)

def parse_args():
    parser = argparse.ArgumentParser(description="Filter initial interpretations based on basic rules and execution results")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Path to input initial interpretations file with SQL predictions")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/initial_interpretations_with_sql_filtered"),
                       help="Directory to save filtered results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for selecting interpretations")
    parser.add_argument("--recompute_metrics", action="store_true",
                       help="Recompute metrics for filtered interpretations")
    return parser.parse_args()

def execute_sql_query(cursor, query: str):
    """Execute SQL query and return results or None if error"""
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except sqlite3.DatabaseError:
        return None

def get_execution_results(db_file: str, sql_queries: List[str]) -> Dict[str, List]:
    """Execute SQL queries and return results"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    results = {}
    for query in sql_queries:
        results[query] = execute_sql_query(cursor, query)
    
    conn.close()
    return results

def filter_interpretations(example: Dict, db_file: str, recompute_metrics: bool = False) -> Dict:
    """Filter interpretations based on SQL query execution results and text criteria"""
    # Get all SQL queries and their interpretations
    all_queries = []
    query_to_interp = {}
    valid_interpretations = []
    
    question_length = len(example["question"].strip())
    
    # First pass: filter invalid interpretations
    for interp in example["interpretations"]:
        # Skip if interpretation text is empty or just whitespace
        # if not interp['interpretation'].strip():
        #     continue
            
        interp_text = interp["interpretation"].strip()
        interp_length = len(interp_text)
        
        # Skip if interpretation contains SQL keywords (likely SQL query)
        sql_keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY", "```sql", "UNION", "INTERSECT", "EXCEPT", "LIMIT", "OFFSET", "AS", "ON", "WITH", "CREATE", "ALTER", "DROP", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "COMMIT", "ROLLBACK", "SAVEPOINT", "BEGIN", "END", "DECLARE", "SET", "EXEC", "EXECUTE", "CALL", "PREPARE", "DEALLOCATE", "EXPLAIN", "ANALYZE", "VACUUM", "REINDEX", "CREATE", "ALTER", "DROP", "RENAME", "ADD", "REMOVE", "MODIFY", "INDEX", "CONSTRAINT", "FOREIGN", "PRIMARY", "UNIQUE", "CHECK", "DEFAULT", "NULL", "NOT", "AND", "OR", "XOR", "IN", "NOT IN", "BETWEEN", "NOT BETWEEN", "LIKE", "NOT LIKE", "GLOB", "REGEXP", "RLIKE", "REGEX", "REGEXMATCH", "REGEXREPLACE", "REGEXEXTRACT", "REGEXEXTRACTALL", "REGEXMATCHES", "REGEXMATCHESALL", "REGEXMATCHESFIRST", "REGEXMATCHESLAST", "REGEXMATCHESPOSITION", "REGEXMATCHESPOSITIONALL", "REGEXMATCHESPOSITIONFIRST", "REGEXMATCHESPOSITIONLAST", "----", " | ", " > ", " < ", " = ", " != ", " >= ", " <= ", " > ", " < ", " >= ", " <= ", " = ", " != ", " AND ", " OR ", " XOR ", " IN ", " NOT IN ", " BETWEEN ", " NOT BETWEEN ", " LIKE ", " NOT LIKE ", " GLOB ", " REGEXP ", " RLIKE ", " REGEX ", " REGEXMATCH ", " REGEXREPLACE ", " REGEXEXTRACT ", " REGEXEXTRACTALL ", " REGEXMATCHES ", " REGEXMATCHESALL ", " REGEXMATCHESFIRST ", " REGEXMATCHESLAST ", " REGEXMATCHESPOSITION ", " REGEXMATCHESPOSITIONALL ", " REGEXMATCHESPOSITIONFIRST ", " REGEXMATCHESPOSITIONLAST ", "*"]
        if any(keyword in interp_text for keyword in sql_keywords):
            # print(f"Skipping interpretation with SQL keywords: {interp_text}")
            continue


        # Check if length differs too much from original question
        if interp_length < question_length/2 :
            # print("\nLength mismatch detected:")
            # print(f"Question ({question_length} chars): {example['question']}")
            # print(f"Interpretation ({interp_length} chars): {interp_text}")
            continue
            
        # If passed all filters, add to valid interpretations
        valid_interpretations.append(interp)
        for query in interp["sql_queries"]:
            all_queries.append(query)
            query_to_interp[query] = interp

    # Execute all queries
    execution_results = get_execution_results(db_file, all_queries)
    
    # Group interpretations by equivalent results
    result_groups = {}
    
    for query, results in execution_results.items():
        if results is None:  # Skip failed queries
            continue
            
        # Find or create a group for this result
        group_found = False
        for group_result in result_groups:
            if compare_query_results(results, group_result, order_by=False):
                result_groups[group_result].append(query_to_interp[query])
                group_found = True
                break
        
        if not group_found:
            result_groups[tuple(map(tuple, results))] = [query_to_interp[query]]
    
    # Select one random interpretation from each group
    selected_interpretations = []
    for group in result_groups.values():
        selected_interpretations.append(random.choice(group))


    # Update example with filtered interpretations
    filtered_example = example.copy()
    filtered_example["interpretations"] = selected_interpretations
    
    # Recompute metrics if requested
    if recompute_metrics:
        try:
            # Collect all SQL queries from filtered interpretations
            all_sql_queries = []
            for interp in selected_interpretations:
                all_sql_queries.extend(interp["sql_queries"])
            
            # Evaluate SQL queries
            sql_metrics = evaluate_predicted_statements(
                db_file,
                all_sql_queries,
                example.get("gold_queries", []),
                remove_duplicates_predictions=False,
                verbose=False,
                return_pred_exec_outputs=True
            )
            
            # Process results
            pred_exec_outputs = remove_duplicate_results(sql_metrics["pred_exec_outputs"])
            num_unique_results = count_unique_results(list(pred_exec_outputs.values()))
            
            # Update metrics in filtered example
            filtered_example["num_unique_execution_results"] = num_unique_results

            filtered_example["interpretation_level_metrics"] = {
                k: v for k, v in sql_metrics.items() 
                if k not in ["pred_exec_outputs", "gold_exec_outputs"]
            }
            
            # Count matched gold SQL queries
            matched_gold_sql = sum(1 for interp in selected_interpretations 
                                 if interp.get("metrics", {}).get("recall", 0) > 0)
            filtered_example["matched_gold_sql"] = matched_gold_sql
            
        except Exception as e:
            print(f"Error computing metrics for question: {example['question']}")
            print(f"Error: {str(e)}")
            filtered_example["num_unique_execution_results"] = 0
            filtered_example["interpretation_level_metrics"] = {}
            filtered_example["matched_gold_sql"] = 0
    
    return filtered_example

def main():
    args = parse_args()
    random.seed(args.seed)
    
    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input data to get dataset type
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    dataset_type = data.get("args", {}).get("dataset_type", "unknown")
    ambrosia_file = data.get("args", {}).get("ambrosia_file", "")
    
    # Handle ambrosia_resplit case
    if dataset_type == "ambrosia" and "resplit" in ambrosia_file:
        dataset_type = "ambrosia_resplit"
    
    # Create model and dataset specific output directory
    model_name = Path(args.input_file).parent.name  # Get model name from parent directory
    model_output_dir = output_dir / model_name / dataset_type
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct output filename
    output_filename = f"{Path(args.input_file).stem}_filtered{Path(args.input_file).suffix}"
    output_path = model_output_dir / output_filename
    
    # Process each example
    filtered_results = []
    metrics_tracker = EvaluationMetricsTracker() if args.recompute_metrics else None
    all_num_unique_results = []
    all_matched_gold_sql = []
    
    print("Filtering interpretations...")

    for example in tqdm(data["results"]):
        filtered_example = filter_interpretations(
            example, 
            example["db_file"],
            recompute_metrics=args.recompute_metrics
        )
        filtered_results.append(filtered_example)
        
        if args.recompute_metrics:
            # Update metrics tracker
            main_key = "ambig" if example.get("is_ambiguous", True) else "unambig"
            ambig_type = example.get("ambig_type", "total")
            
            if filtered_example.get("interpretation_level_metrics"):
                metrics_tracker.update_metrics(
                    main_key, 
                    ambig_type, 
                    filtered_example["interpretation_level_metrics"]
                )
                all_num_unique_results.append(filtered_example["num_unique_execution_results"])
                all_matched_gold_sql.append(filtered_example["matched_gold_sql"])
            else:
                metrics_tracker.add_zero_metrics(main_key, ambig_type)
                all_num_unique_results.append(0)
                all_matched_gold_sql.append(0)
    
    # Update results
    filtered_data = data.copy()
    filtered_data["results"] = filtered_results
    filtered_data["args"]["filtered"] = True
    
    if args.recompute_metrics:
        filtered_data["metrics"] = metrics_tracker.get_aggregated_metrics()
        filtered_data["statistics"] = {
            "total_examples": len(filtered_results),
            "avg_interpretations": np.mean([len(ex["interpretations"]) for ex in filtered_results]),
            "avg_unique_execution_results": np.mean(all_num_unique_results),
            "avg_matched_gold_sql": np.mean(all_matched_gold_sql)
        }
    
    # Save filtered results
    print(f"Saving filtered results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    # Print statistics
    original_interp_count = sum(len(ex["interpretations"]) for ex in data["results"])
    filtered_interp_count = sum(len(ex["interpretations"]) for ex in filtered_results)
    
    print("\nFiltering Statistics:")
    print(f"Original interpretations: {original_interp_count}")
    print(f"Filtered interpretations: {filtered_interp_count}")
    print(f"Reduction: {(original_interp_count - filtered_interp_count) / original_interp_count * 100:.2f}%")
    
    if args.recompute_metrics:
        print("\nUpdated Metrics:")
        metrics_tracker.print_summary(short=True)

if __name__ == "__main__":
    main() 