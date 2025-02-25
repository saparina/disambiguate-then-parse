from typing import Dict, List
from utils.model_utils import generate_from_prompt
from utils.metrics import evaluate_predicted_statements
from utils.output_parsers import parse_statements_llama
from prompts import user_message_sql

def generate_and_evaluate_sql(
    model,
    tokenizer,
    db_dump: str,
    text: str,  # Can be either question or interpretation
    db_file: str = None,
    gold_queries: List[str] = None,
    generation_config: Dict = None,
    prompt_template: str = None,
    verbose: bool = False,
    return_pred_exec_outputs: bool = False
) -> Dict:
    """Generate SQL queries for given text and evaluate if gold queries provided"""
    
    # Default to standard SQL prompt if none provided
    if prompt_template is None:
        prompt_template = user_message_sql
        
    messages = [{
        "role": "user", 
        "content": prompt_template.format(db_dump, text)
    }]
    
    sql_prediction = generate_from_prompt(
        model, tokenizer, messages, generation_config, max_length=8192
    )

    if verbose:
        print(f"SQL prediction: {sql_prediction}\n")

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
    
    if verbose:
        print("Parsed SQL queries:")
        print('\n\n'.join(sql_queries))
        print()
        print("Gold queries:")
        print('\n\n'.join(gold_queries))
        print()

    result = {
        "success": True,
        "sql_queries": sql_queries,
        "original_prediction": sql_prediction,
        "metrics": None
    }
    
    # Add metrics if gold queries and db_file available
    if gold_queries and db_file:
        try:
            result["metrics"] = evaluate_predicted_statements(
                db_file,
                sql_queries,
                gold_queries,
                remove_duplicates_predictions=False,
                verbose=verbose,
                return_pred_exec_outputs=return_pred_exec_outputs
            )
        except Exception as e:
            result["metrics"] = {
                "error": str(e),
                "recall": 0.0,
                "precision": 0.0,
                "f1_score": 0.0
            }
    
    return result 