import pandas as pd
import json
from typing import Union, Tuple
from datasets import Dataset, concatenate_datasets
from .ambiqt_reader import fix_dataset_dbs, read_ambiqt
from .config import DataConfig, TaskType
from .utils import (
    filter_gold,
    filter_interpr,
    add_nl_interpretations,
    load_all_sql_interpretations
)
from .db_utils import (
    merge_all_insert_statements
)
from .utils import logger, log_dataset_info, merge_interpretations
import random

def load_ambrosia(config: DataConfig) -> pd.DataFrame:
    """Load and preprocess Ambrosia dataset"""
    df = pd.read_csv(config.ambrosia_file)
    
    # Basic preprocessing
    df['gold_queries'] = df['gold_queries'].str.split('\n\n')

    # Fix path to db file
    df['db_file'] = df['db_file'].str.replace('data/', config.data_dir + '/ambrosia/data/')

    # Handle attachment type questions
    attachment_mask = (df['ambig_type'] == 'attachment') & (~df['question'].str.endswith(" Show them in one table."))
    df.loc[attachment_mask, 'question'] += " Show them in one table."
    df.loc[attachment_mask, 'ambig_question'] += " Show them in one table."
    
    # Process database dumps
    df['db_dump'] = df.apply(lambda row: merge_all_insert_statements(row['db_file'], row['db_dump']), axis=1)
    
    # Add interpretations
    df = add_nl_interpretations(df)
    
    # Filter by split if not loading all data
    if config.split != "all":
        if config.split == "train":
            if config.validation:
                df = df[df['split'].isin(["few_shot_examples", "validation"])]
            else:
                df = df[df['split'] == "few_shot_examples"]
        else:
            df = df[df['split'] == config.split]
    
    df['dataset_source'] = 'ambrosia'
    return df

def load_ambiqt(config: DataConfig) -> pd.DataFrame:
    """Load and preprocess AmbiQT dataset"""

    if config.ambiqt_interpr_file: #and (config.gold_interpr or config.learn_missing_interpr or config.learn_gold_interpr):
        # Load from interpretation file
        data = json.load(open(config.ambiqt_interpr_file, "r"))
        data = data[2:]  # Skip first two examples as per original code
        
        if config.filter_interpr:
            data = filter_interpr(data)
            
        df = pd.DataFrame(data)
        
        # Process interpretations
        df["nl_interpretations"] = df["interpretations"]
        df["nl_interpretations"] = df.apply(
            lambda row: '\n\n'.join([x[0].replace("_", " ") for x in row["nl_interpretations"]]), 
            axis=1
        )
        df = df.drop('interpretations', axis=1)
    else:
        # Load raw data
        if config.split == "test" or config.split == "validation":
            syn_types = ["column", "table", "split", "agg"]
        else:
            syn_types = ["column", "table"]

        data = read_ambiqt(config.data_dir, split="validation" if config.split == "test" else config.split, syn_types=syn_types)
        data = fix_dataset_dbs(data)

        if config.filter_gold:
            data = filter_gold(data)
            
        df = pd.DataFrame(data)
    
    # Common processing
    df["db_dump_processed"] = df["db_dump"]
    df["ambig_type"] = "vague"
    df["question_type"] = "ambig"
    df["ambig_question"] = df["question"]
    df['split'] = config.split if config.split != "validation" else "test"
    
    df['dataset_source'] = 'ambiqt'
    return df

def balance_dataset(dataset, balance_factors=['dataset_source', 'is_ambiguous', 'interpretation_model']):
    """
    Balance dataset by specified factors using weighted sampling.
    Balances in order of priority:
    1. dataset_source (ambrosia/ambiqt) - maintain roughly equal representation
    2. is_ambiguous (True/False) - within each dataset source
    3. interpretation_model - within each ambiguity class
    
    Returns a new balanced dataset.
    """
    df = dataset.to_pandas()

    # Step 1: Balance dataset sources first
    if 'dataset_source' in balance_factors and 'dataset_source' in df.columns:
        ambrosia_size = len(df[df['dataset_source'] == 'ambrosia'])
        ambiqt_size = len(df[df['dataset_source'] == 'ambiqt'])
        
        # Calculate multiplication factor based on ratio
        mult = round(ambiqt_size / ambrosia_size)
        
        balanced_sources = []
        for source in df['dataset_source'].unique():
            source_df = df[df['dataset_source'] == source]
            if source == 'ambrosia':  # Upsample the smaller dataset
                balanced_sources.append(pd.concat([source_df] * mult, ignore_index=True))
            else:
                balanced_sources.append(source_df)
        
        df = pd.concat(balanced_sources, ignore_index=True)

    logger.info("Dataset balancing statistics:")
    for factor in balance_factors:
        if factor not in df.columns:
            logger.info(f"\n{factor} not found in dataset")
            continue
            
        logger.info(f"\n{factor} distribution:")
        if factor == 'dataset_source':
            logger.info(df[factor].value_counts())
        else:
            for source in df['dataset_source'].unique():
                logger.info(f"\nIn {source}:")
                logger.info(df[df['dataset_source'] == source][factor].value_counts())
    return Dataset.from_pandas(df)

def load_dataset(config: DataConfig) -> Union[Dataset, Tuple[Dataset, Dataset]]:
    """
    Main function to load and process dataset
    Returns either a single dataset or a tuple of (train, validation) datasets
    """
    additional_info = {
        "filter_gold": config.filter_gold,
        "filter_interpr": config.filter_interpr,
        "sample_size": config.sample_size,
        "ambrosia_question_type": config.ambrosia_question_type
    }
    
    log_dataset_info(
        dataset_type=config.dataset_type,
        split=config.split,
        task=config.task_type.value if config.task_type else "None",
        num_examples=None,  # Will be updated after loading
        additional_info=additional_info
    )
    
    dfs = []
    
    # Load datasets based on config
    if config.dataset_type in ["ambrosia", "all"]:
        df_ambrosia = load_ambrosia(config)
        
        # Filter by question type if specified
        if config.ambrosia_question_type:
            is_ambig = config.ambrosia_question_type == "ambig"
            df_ambrosia = df_ambrosia[df_ambrosia['is_ambiguous'] == is_ambig]
            logger.info(f"Filtered Ambrosia dataset to {config.ambrosia_question_type} questions: {len(df_ambrosia)} examples")

        if config.ambrosia_question_type_test:
            is_ambig = config.ambrosia_question_type_test == "ambig"
            df_ambrosia = df_ambrosia[df_ambrosia['is_ambiguous'] == is_ambig]
            logger.info(f"Filtered Ambrosia dataset to {config.ambrosia_question_type_test} questions: {len(df_ambrosia)} examples")
            
        dfs.append(df_ambrosia)
        
    if config.dataset_type in ["ambiqt", "all"]:
        df_ambiqt = load_ambiqt(config)
        dfs.append(df_ambiqt)
    
    # Combine datasets if needed
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    
    # Handle predicted interpretations for learn_missing_interpr option
    if config.learn_missing_interpr:
        sql_output_dir = config.sql_output_dir
        
        if config.dataset_type == "all":
            # Load interpretations for both datasets
            df_interpr_ambrosia = load_all_sql_interpretations(
                output_dir=sql_output_dir,
                dataset_type="ambrosia",
                split=config.split
            )
            df_interpr_ambiqt = load_all_sql_interpretations(
                output_dir=sql_output_dir,
                dataset_type="ambiqt",
                split=config.split if config.split != "validation" else "test"
            )
            
            # Combine interpretations if both are available
            if df_interpr_ambrosia is not None and df_interpr_ambiqt is not None:
                df_interpr = pd.concat([df_interpr_ambrosia, df_interpr_ambiqt], ignore_index=True)
            elif df_interpr_ambrosia is not None:
                df_interpr = df_interpr_ambrosia
            elif df_interpr_ambiqt is not None:
                df_interpr = df_interpr_ambiqt
            else:
                df_interpr = None
        else:
            if config.split == "validation" and config.dataset_type == "ambiqt":
                split = "test"
            else:
                split = config.split
            
            # Load interpretations for single dataset
            df_interpr = load_all_sql_interpretations(
                output_dir=sql_output_dir,
                dataset_type=config.dataset_type,
                split=split
            )
        
        if df_interpr is not None:
            df = merge_interpretations(df, df_interpr)
            logger.info(f"Added interpretations from multiple models. New dataset size: {len(df)}")

            # Calculate number of rows with all interpretations covered if column exists
            if 'missing_nl_interpretations' in df.columns:
                num_covered = len(df[df['missing_nl_interpretations'] == "All possible interpretations are covered"])
                logger.info(f"Number of rows with all interpretations covered: {num_covered}")

            if 'found_interpretations' in df.columns:
                num_covered = len(df[df['found_interpretations'] == "No interpretations are covered"])
                logger.info(f"Number of rows with no interpretations found: {num_covered}")

            if 'corrected_nl_interpretations' in df.columns:
                num_covered = len(df[df['corrected_nl_interpretations'].str.startswith("No interpretations")])
                if num_covered > 0:
                    import pdb; pdb.set_trace()
                # logger.info(f"Number of rows with all interpretations covered: {num_covered}")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    
    # Apply sampling if configured
    if config.sample_size:
        dataset = dataset.select(range(min(config.sample_size, len(dataset))))
        logger.info(f"Sampled {config.sample_size} examples from dataset")

    # Only split AmbiQT data for finetuning training data
    if config.task_type == TaskType.FINETUNING and config.split == "train" and config.dataset_type in ["ambiqt", "all", "ambrosia"]:
        if config.interpretation_model_train and 'initial_generated_interpr' in dataset.features:
            dataset = dataset.to_pandas()
            dataset = dataset[dataset['interpretation_model'] == config.interpretation_model_train].reset_index(drop=True)
            dataset = Dataset.from_pandas(dataset)

        if config.dataset_type == "all" or config.dataset_type == "ambrosia":
            # Split and balance datasets
            ambrosia_dataset = dataset.filter(lambda x: x['dataset_source'] == 'ambrosia')
            ambrosia_dataset = ambrosia_dataset.shuffle(seed=42).flatten_indices()

            # Split Ambrosia portion
            ambrosia_train = ambrosia_dataset.filter(lambda x: x['split'] == 'few_shot_examples')
            ambrosia_val = ambrosia_dataset.filter(lambda x: x['split'] == 'validation')
            
            # Deduplicate interpretations in validation sets
            # Group by question and db_file, randomly select one interpretation for Ambrosia
            ambrosia_val_df = ambrosia_val.to_pandas()
            ambrosia_val_df = ambrosia_val_df.groupby(['question', 'db_file']).sample(n=1, random_state=42).reset_index(drop=True)
            ambrosia_val = Dataset.from_pandas(ambrosia_val_df)
            
        if config.dataset_type == "all" or config.dataset_type == "ambiqt":
            # Split ambiqt portion
            ambiqt_dataset = dataset.filter(lambda x: x['dataset_source'] == 'ambiqt')
            ambiqt_dataset = ambiqt_dataset.shuffle(seed=42).flatten_indices()
            
            # For ambiqt, split based on database files and deduplicate
            unique_dbs = list(set(ambiqt_dataset['db_file']))
            num_val_dbs = min(100, len(unique_dbs))  # ~100 databases for validation
            
            # Randomly select databases for validation
            val_dbs = set(random.sample(unique_dbs, num_val_dbs))
            
            # Split based on databases
            ambiqt_train = ambiqt_dataset.filter(lambda x: x['db_file'] not in val_dbs)
            ambiqt_val = ambiqt_dataset.filter(lambda x: x['db_file'] in val_dbs)

            # Deduplicate ambiqt validation set
            ambiqt_val_df = ambiqt_val.to_pandas()
            ambiqt_val_df = ambiqt_val_df.groupby(['question', 'db_file']).sample(n=1, random_state=42).reset_index(drop=True)
            ambiqt_val = Dataset.from_pandas(ambiqt_val_df)

        if config.dataset_type == "all":    
            train_combined = concatenate_datasets([ambiqt_train, ambrosia_train])
            val_combined = concatenate_datasets([ambiqt_val, ambrosia_val])
             
            # Balance training data if configured
            if config.balance_dataset:   
                train_balanced = balance_dataset(
                    train_combined, 
                    balance_factors=config.balance_factors
                )
                val_balanced = balance_dataset(
                    val_combined, 
                    balance_factors=config.balance_factors
                )
                return train_balanced, val_balanced
            else:
                return train_combined, val_combined
            
        elif config.dataset_type == "ambrosia":                
            return ambrosia_train, ambrosia_val
        else:
            return ambiqt_train, ambiqt_val
        
    if config.task_type == TaskType.FINETUNING and (config.split == "test" or config.split == "validation" and config.dataset_type == "ambiqt"):
        dataset = dataset.to_pandas()
        if config.interpretation_model_test and 'initial_generated_interpr' in dataset.columns:
            dataset = dataset[dataset['interpretation_model'] == config.interpretation_model_test].reset_index(drop=True)
        dataset = dataset.groupby(['question', 'db_file']).sample(n=1, random_state=42).reset_index(drop=True)
        dataset = Dataset.from_pandas(dataset)
    
    # Update logging with actual number of examples
    if isinstance(dataset, tuple):
        log_dataset_info(
            dataset_type=config.dataset_type,
            split="train",
            task=config.task_type.value if config.task_type else "None",
            num_examples=len(dataset[0])
        )
        log_dataset_info(
            dataset_type=config.dataset_type,
            split="val",
            task=config.task_type.value if config.task_type else "None",
            num_examples=len(dataset[1])
        )
    else:
        log_dataset_info(
            dataset_type=config.dataset_type,
            split=config.split,
            task=config.task_type.value if config.task_type else "None",
            num_examples=len(dataset)
        )
    
    return dataset
