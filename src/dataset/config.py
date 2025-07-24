from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
import yaml

class TaskType(Enum):
    GENERATE_INTERPRETATIONS = "generate_interpretations"
    FINETUNING = "finetuning"
    TEXT2SQL_BASELINE = "text2sql_baseline"

@dataclass
class DataConfig:
    """Configuration for dataset loading"""
    # Required settings
    dataset_type: str  # "ambrosia", "ambiqt", or "all"
    split: str  # "train", "validation", "test", "few_shot_examples"
    task_type: Optional[TaskType] = None  # Keep TaskType
    
    # Paths
    ambrosia_file: str = "data/ambrosia/data/ambrosia_resplit.csv"
    data_dir: str = "data/"
    
    # Filtering options
    filter_gold: bool = False
    filter_interpr: bool = False
    sample_size: Optional[int] = None
    ambrosia_question_type: Optional[str] = None
    ambrosia_question_type_test: Optional[str] = None
    # Interpretation related
    ambiqt_interpr_file: Optional[str] = None
    gold_interpr: bool = False
    learn_missing_interpr: bool = False
    learn_gold_interpr: bool = False
    # Additional options
    unambig_sql: bool = False
    lf_type: Optional[str] = None
    icl_examples: int = 0
    validation: bool = False
    balance_dataset: bool = False  # Whether to balance dataset during loading
    balance_factors: List[str] = field(
        default_factory=lambda: ['dataset_source', 'is_ambiguous', 'interpretation_model']
    )
    use_balanced_sampler: bool = False  # Whether to use balanced sampling during training
    interpretation_model_train: Optional[str] = None
    interpretation_model_test: Optional[str] = None
    sql_output_dir: Optional[str] = None

def load_dataset_config(
    dataset_type: str,
    split: str,
    task_type: Optional[TaskType] = None,
    config_path: str = "src/configs/dataset_configs.yaml",
    sample_size: Optional[int] = None,
    ambrosia_question_type: Optional[str] = None,
    ambrosia_question_type_test: Optional[str] = None,
    learn_missing_interpr: bool = False,
    learn_gold_interpr: bool = False,
    balance_dataset: bool = False,
    interpretation_model_train: Optional[str] = None,
    interpretation_model_test: Optional[str] = None,
    sql_output_dir: Optional[str] = None,
    **kwargs
) -> DataConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Get config key based on task type
    if task_type == TaskType.FINETUNING:
        config_key = "finetuning"
    elif task_type == TaskType.TEXT2SQL_BASELINE:
        config_key = "text2sql_baseline"
    elif task_type == TaskType.GENERATE_INTERPRETATIONS:
        config_key = "generate_interpretations"
    else:
        config_key = "default"
    
    dataset_config = config_dict[config_key][split].copy()
    
    # Add required fields
    dataset_config['dataset_type'] = dataset_type
    dataset_config['split'] = split
    dataset_config['task_type'] = task_type
    dataset_config['sample_size'] = sample_size
    dataset_config['ambrosia_question_type'] = ambrosia_question_type
    dataset_config['ambrosia_question_type_test'] = ambrosia_question_type_test
    dataset_config['learn_missing_interpr'] = learn_missing_interpr
    dataset_config['learn_gold_interpr'] = learn_gold_interpr
    dataset_config['balance_dataset'] = balance_dataset
    dataset_config['interpretation_model_test'] = interpretation_model_test
    dataset_config['interpretation_model_train'] = interpretation_model_train
    dataset_config['sql_output_dir'] = sql_output_dir
    if 'ambrosia_file' in kwargs:
        dataset_config['ambrosia_file'] = kwargs['ambrosia_file']
    if 'data_dir' in kwargs:
        dataset_config['data_dir'] = kwargs['data_dir']

    return DataConfig(**dataset_config)