import re
import os
import json
import pandas as pd
import sqlite3
from typing import Dict, Optional, Any
import sys
sys.path.append("..")  # Add parent directory to path
from utils import duplicate_exact, evaluate_predicted_statements
import logging
import wandb
from tqdm import tqdm
from pathlib import Path


def append_string_to_file(text, file_path):
  with open(file_path, 'a') as file:
      file.write(text + '\n')

def remove_spaces(text):
  return re.sub(r'\s+', ' ', text)

def transform_gold_queries(row):
    if row['is_ambiguous']:
        if row['ambig_type'] == 'attachment' and isinstance(row['gold_queries'], list) and len(row['gold_queries']) > 1:
            row['gold_queries'] = [row['gold_queries'][1]]
        elif row['ambig_type'] == 'scope' and isinstance(row['gold_queries'], list) and len(row['gold_queries']) > 0:
            row['gold_queries'] = [row['gold_queries'][0]]
        elif row['ambig_type'] == 'vague' and isinstance(row['gold_queries'], list) and len(row['gold_queries']) > 2:
            row['gold_queries'] = [row['gold_queries'][2]]
        else:
            return None  # Skip this example if vague and third element does not exist
    return row

def create_ambig_to_unambig_mapping(df):
    ambig_to_unambig = {}
    for _, row in df.iterrows():
        if row['ambig_question'] and row['ambig_question'] != row['question']:
            if row['ambig_question'] not in ambig_to_unambig:
                ambig_to_unambig[row['ambig_question']] = []
            ambig_to_unambig[row['ambig_question']].append(row['question'])
    return ambig_to_unambig

def add_nl_interpretations(df: pd.DataFrame) -> pd.DataFrame:
    """Add natural language interpretations to the dataframe"""
    ambig_to_unambig = create_ambig_to_unambig_mapping(df)

    def process_row(row):
        if row['is_ambiguous']:
            interpretations = ambig_to_unambig.get(row['question'], ["No matching interpretation found"])
            return '\n\n'.join(interpretations)
        else:
            return row['question']

    df['nl_interpretations'] = df.apply(process_row, axis=1)
    return df

def add_interpretation_rows(df):
    new_rows = []

    for _, row in df.iterrows():
        if row['is_ambiguous']:
            if pd.notnull(row['nl_interpretations']):
                interpretations = row['nl_interpretations'].split('\n\n')
                gold_queries = row['gold_queries']

                for interpretation, gold_query in zip(interpretations, gold_queries):
                    # Create a new row for each interpretation
                    new_row = row.copy()
                    new_row['question'] = interpretation
                    new_row['is_ambiguous'] = False
                    new_row['question_type'] = "unambig"
                    new_row['gold_queries'] = [gold_query]
                    new_rows.append(new_row)

    expanded_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return expanded_df

def filter_gold(eval_dataset):
    final_eval_dataset = []
    for example in eval_dataset:
        conn = sqlite3.connect(example['db_file'])
        cursor = conn.cursor()
        filtered = False

        results = []
        for query in example['gold_queries']:
            try:
                cursor.execute(query)
                res = cursor.fetchall()
                results.append(res)
            except Exception as e:
                filtered = True
                break

            if not res and example['syn_type'] not in ["split", "agg"]:
                filtered = True
                break

        
        if not filtered:
            has_duplicates, _, _ = duplicate_exact(results)
            if not has_duplicates:
                final_eval_dataset.append(example)

        conn.close()

    print(f"Original dataset {len(eval_dataset)}")
    print(f"Filtered dataset {len(final_eval_dataset)}")
    return final_eval_dataset

def filter_interpr(eval_dataset):
    final_eval_dataset = []
    for example in eval_dataset:
        if example["interpretations"] and sum(interp[1] for interp in example["interpretations"]) == len(example["interpretations"]):
            final_eval_dataset.append(example)

    print(f"Original interpr {len(eval_dataset)}")
    print(f"Filtered interpr {len(final_eval_dataset)}")
    return final_eval_dataset


def evaluate_missing_gold_queries(entry):
    """
    Evaluate which gold queries are not covered by the predicted interpretations.
    Returns missing and found gold queries.
    """
    # Get all SQL queries from all interpretations
    sql_predictions = []
    for interp in entry['generated_interpretations']:
        if interp['metrics']['recall'] > 0:
            sql_predictions.extend(interp['sql_queries'])
    
    # Track which gold queries are covered
    all_gold_queries = {x: 0 for x in entry['gold_queries']}
    
    # Check each prediction against gold queries
    for pred_query in sql_predictions:
        for gold_query in entry['gold_queries']:
            results = evaluate_predicted_statements(
                entry['db_file'], [pred_query], [gold_query],
                remove_duplicates_predictions=False, verbose=False
            )
            if results['f1_score'] == 1:
                all_gold_queries[gold_query] += 1
                break
    
    missing_gold_queries = []
    found_gold_queries = []
    for query, count in all_gold_queries.items():
        if count == 0:
            missing_gold_queries.append(query)
        else:
            found_gold_queries.append(query)
            
    return missing_gold_queries, found_gold_queries

def get_missing_nl_interpretations(row):
    """
    Map missing gold queries to their corresponding NL interpretations.
    Returns a string of missing interpretations joined by newlines.
    """
    if not row.get('nl_interpretations'):
        return "No interpretations available"
        
    # Split the nl_interpretations into a list
    nl_interpretations_list = row['nl_interpretations'].split('\n\n')
    
    # Map missing gold queries to their corresponding NL interpretations
    missing_nl = []
    for missing_query in row['missing_gold_queries']:
        try:
            # Get the index of the missing query in the original gold_queries
            index = row['gold_queries'].index(missing_query)
            # Use the index to extract the corresponding NL interpretation
            if index < len(nl_interpretations_list):
                missing_nl.append(nl_interpretations_list[index])
        except ValueError:
            continue
    
    return '\n'.join(missing_nl) if missing_nl else "All possible interpretations are covered"

def get_found_nl_interpretations(row):
    """
    Map found gold queries to their corresponding NL interpretations.
    Returns a string of found interpretations joined by newlines.
    """
    if not row.get('nl_interpretations'):
        return "No interpretations available"
        
    
    # Track which gold queries have been matched and their corresponding interpretations
    gold_query_matches = {}  # Maps gold query to the first matching interpretation
    
    # Check each interpretation
    for interp in row['generated_interpretations']:
        if interp['metrics']['recall'] > 0:
            # For each SQL query in this interpretation
            for pred_query in interp['sql_queries']:
                # Check against each gold query
                for gold_query in row['gold_queries']:
                    # Skip if we already found a match for this gold query
                    if gold_query in gold_query_matches:
                        continue
                        
                    results = evaluate_predicted_statements(
                        row['db_file'], [pred_query], [gold_query],
                        remove_duplicates_predictions=False, verbose=False
                    )
                    
                    # If perfect match and we haven't stored this gold query yet
                    if results['f1_score'] == 1:
                        # Store the interpretation for this gold query
                        gold_query_matches[gold_query] = interp['interpretation']
                        break
    
    # Get the unique interpretations that matched
    found_interpretations = list(set(gold_query_matches.values()))
    
    return '\n'.join(found_interpretations) if found_interpretations else "No interpretations are covered"


def merge_interpretations(df: pd.DataFrame, df_interpr: pd.DataFrame) -> pd.DataFrame:
    """Merge interpretations with main dataset"""
    # Create expanded dataframe with separate rows for each question/db_file
    expanded_rows = []
    
    for _, row in tqdm(df.iterrows()):
        # Find matching interpretations for this question/db_file
        matching_interpr = df_interpr[
            (df_interpr['db_file'] == row['db_file']) & 
            (df_interpr['question'] == row['question'])
        ]

        try:
            if not matching_interpr.empty:
                
                # Add all matching rows since we want all interpretations
                for _, interp_row in matching_interpr.iterrows():
                    new_row = row.copy()
                    new_row['interpretation_model'] = interp_row['interpretation_model']
                    new_row['sql_model'] = interp_row['sql_model']
                    new_row['generated_interpretations'] = interp_row['generated_interpretations']
                    for interp_idx, interp in enumerate(new_row['generated_interpretations']):
                        if "metrics" in interp and "execution_errors" in interp["metrics"]:
                            new_row['generated_interpretations'][interp_idx]["metrics"]["execution_errors"] = []

                    new_row['initial_generated_interpr'] = [interp['interpretation'] for interp in interp_row['generated_interpretations']]
                    if not new_row['initial_generated_interpr']:
                        new_row['initial_generated_interpr'] = ["No existing interpretations available"]

                    if 'nl_interpretations' in row and row['nl_interpretations'] and row['split'] != "test":
                        # Find missing gold queries and their interpretations
                        new_row['missing_gold_queries'], new_row['found_gold_queries'] = evaluate_missing_gold_queries(new_row)

                        new_row['missing_nl_interpretations'] = get_missing_nl_interpretations(new_row)
                        new_row['found_nl_interpretations'] = get_found_nl_interpretations(new_row)

                        corrected_nl_interpretations = []
                        if not new_row["found_nl_interpretations"].startswith("No interpretations "):
                            corrected_nl_interpretations.append(new_row["found_nl_interpretations"])
                        if not new_row["missing_nl_interpretations"].startswith("All possible interpretations are covered"):
                            corrected_nl_interpretations.append(new_row["missing_nl_interpretations"])
                        new_row["corrected_nl_interpretations"] = "\n".join(corrected_nl_interpretations)
                    
                    expanded_rows.append(new_row)
            else:
                continue
        except Exception as e:
            print(f"Error processing row {row['question']}: {e}")
            import pdb; pdb.set_trace()
    return pd.DataFrame(expanded_rows)

def read_predicted_interpretations(json_file: str) -> pd.DataFrame:
    """Read and process SQL generation results file"""
    if not os.path.exists(json_file):
        logger.warning(f"File '{json_file}' not found.")
        return None
    
    logger.info(f"Reading {json_file}")
        
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Extract model names from filename
    filename = os.path.basename(json_file)
    try:
        interp_model = filename.split('_interp_')[1].split('_sql_')[0]
        sql_model = filename.split('_sql_')[1].split('_seed')[0]
    except IndexError:
        logger.warning(f"Could not extract model names from filename: {filename}")
        return None
    
    # Process results and group interpretations by question
    results = []
    for entry in tqdm(data['results']):
        # Create a single result with all interpretations
        result = {
            'db_file': entry['db_file'],
            'db_dump': entry['db_dump'],
            'question': entry['question'],
            'gold_queries': entry['gold_queries'],
            'interpretation_model': interp_model,
            'sql_model': sql_model,
            'generated_interpretations': [{
                'interpretation': interp['interpretation'],
                'sql_queries': interp['sql_queries'],
                'metrics': interp['metrics']
            } for interp in entry['interpretations']]
        }
        results.append(result)
    
    return pd.DataFrame(results)

def load_all_sql_interpretations(output_dir: str, dataset_type: str, split: str) -> pd.DataFrame:
    """Load all SQL interpretation files for a given dataset type and split"""
    # Define patterns for both training and validation files
    patterns = [
        f"sql_interp_*_{dataset_type}_{split}_*.json",  # Standard pattern
        f"sql_interp_*_{dataset_type}_validation_*.json" if split == "train" else None,  # Validation pattern
    ]
    
    # Collect all matching files
    dfs = []
    for pattern in patterns:
        if not pattern:
            continue
        
        files = list(Path(output_dir).glob(pattern))
        if not files:
            logger.info(f"No interpretation files found matching pattern: {pattern}")
            continue
            
        # Read and combine all matching files
        for file in files:
            df = read_predicted_interpretations(str(file))
            if df is not None:
                logger.info(f"Loaded interpretations from {file}")
                dfs.append(df)
    
    if not dfs:
        logger.warning(f"No interpretation files found for {dataset_type}/{split} in {output_dir}")
        return None
        
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(dfs)} interpretation files with total {len(combined_df)} rows")
    return combined_df

def setup_logger(use_wandb: bool = False, wandb_config: Optional[Dict[str, Any]] = None):
    """Setup basic logging configuration with optional wandb integration"""
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize wandb if requested
    if use_wandb and wandb_config:
        wandb.init(**wandb_config)
        
        # Create a custom handler that logs to both console and wandb
        class WandbHandler(logging.Handler):
            def emit(self, record):
                msg = self.format(record)
                wandb.log({"log": msg})
        
        wandb_handler = WandbHandler()
        logger.addHandler(wandb_handler)
    
    return logger

def log_dataset_info(dataset_type: str, split: str, task: str, num_examples: int, 
                    additional_info: Optional[Dict[str, Any]] = None):
    """Log dataset information to both console and wandb if enabled"""
    info = {
        "dataset_type": dataset_type,
        "split": split,
        "task": task,
        "num_examples": num_examples
    }
    if additional_info:
        info.update(additional_info)
    
    # Log to console
    logger.info(f"Dataset Info:")
    for k, v in info.items():
        logger.info(f"  {k}: {v}")
    
    # Log to wandb if initialized
    if wandb.run is not None:
        wandb.log({f"dataset/{k}": v for k, v in info.items()})

logger = setup_logger()