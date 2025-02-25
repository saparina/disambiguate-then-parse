from datasets import Dataset
from .config import TaskType, load_dataset_config
from .loaders import load_dataset
from .utils import logger

def map_logical_to_file_split(dataset_type: str, logical_split: str, task_type: TaskType) -> str:
    """Maps logical splits to actual file splits based on dataset and task"""
    if dataset_type == "ambiqt":  # AmbiQT
        if logical_split == "test":
            return "validation"
        else:
            return "train"
    else:
        return logical_split

def load_baseline_dataset(dataset_type: str, for_train: bool = False, **kwargs) -> Dataset:
    """Load dataset specifically for baseline evaluation"""
    logger.info(f"Loading {dataset_type} dataset for {'ICL' if for_train else 'baseline evaluation'}")
    
    ambrosia_question_type = kwargs.pop('ambrosia_question_type', None)
    
    config = load_dataset_config(
        dataset_type=dataset_type,
        split="train" if for_train else "test",
        task_type=TaskType.TEXT2SQL_BASELINE,
        ambrosia_question_type=ambrosia_question_type,
        ambrosia_file=kwargs.pop('ambrosia_file', None),
        **kwargs  # Pass through any additional config options
    )
    
    config.split = map_logical_to_file_split(
        dataset_type, 
        "train" if for_train else "test",
        TaskType.TEXT2SQL_BASELINE
    )
    
    dataset = load_dataset(config)
    logger.info(f"Loaded {len(dataset)} examples from {dataset_type}")
    return dataset

def load_finetuning_datasets(args):
    """Load datasets for finetuning"""
    logger.info(f"Loading {args.dataset_type} datasets for finetuning")
    
    train_config = load_dataset_config(
        dataset_type=args.dataset_type,
        split="train",
        task_type=TaskType.FINETUNING,
        sql_output_dir=args.sql_output_dir,
        learn_missing_interpr=args.learn_missing_interpr,
        learn_gold_interpr=args.learn_gold_interpr,
        balance_dataset=args.balance_dataset,
        ambrosia_question_type=args.question_type,
        interpretation_model_train=args.interpretation_model_train,
        interpretation_model_test=args.interpretation_model_test,
        ambrosia_file=args.ambrosia_file
    )
    
    train_config.split = map_logical_to_file_split(
        args.dataset_type, 
        "train",
        TaskType.FINETUNING
    )
    train_dataset, val_dataset = load_dataset(train_config)
    logger.info(f"Loaded {len(train_dataset)} training examples")
    logger.info(f"Loaded {len(val_dataset)} validation examples")
    
    test_config = load_dataset_config(
        dataset_type=args.dataset_type,
        split="test",
        task_type=TaskType.FINETUNING,
        learn_missing_interpr=args.learn_missing_interpr,
        learn_gold_interpr=args.learn_gold_interpr,
        balance_dataset=args.balance_dataset,
        interpretation_model_test=args.interpretation_model_test,
        sql_output_dir=args.sql_output_dir,
        ambrosia_file=args.ambrosia_file,
        ambrosia_question_type_test=args.question_type_test
    )
    
    test_config.split = map_logical_to_file_split(
        args.dataset_type, 
        "test",
        TaskType.FINETUNING
    )

    test_dataset = load_dataset(test_config)
    logger.info(f"Loaded {len(test_dataset)} test examples")
    
    return train_dataset, val_dataset, test_dataset

def load_interpretations_dataset(args) -> Dataset:
    """Load dataset for interpretation generation"""
    logger.info(f"Loading {args.dataset_type} dataset for interpretation generation")
    logger.info(f"Split: {args.split}")
    
    config = load_dataset_config(
        dataset_type=args.dataset_type,
        split=args.split,
        task_type=TaskType.GENERATE_INTERPRETATIONS,
        ambrosia_file=getattr(args, 'ambrosia_file', None),
        ambrosia_question_type_test=getattr(args, 'ambrosia_question_type_test', None)
    )
    
    config.split = map_logical_to_file_split(
        args.dataset_type, 
        config.split,
        TaskType.GENERATE_INTERPRETATIONS
    )
    
    dataset = load_dataset(config)
    logger.info(f"Loaded {len(dataset)} examples")
    
    return dataset 