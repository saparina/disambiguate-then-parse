import numpy as np

class EvaluationMetricsTracker:
    """Tracker for evaluation metrics across different types of questions"""
    
    def __init__(self):
        self.metrics = {
            "ambig": {"attachment": {}, "scope": {}, "vague": {}, "total": {}},
            "unambig": {"attachment": {}, "scope": {}, "vague": {}, "total": {}}
        }
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize metric lists for all categories"""
        for main_key in self.metrics:
            for ambig_type in self.metrics[main_key]:
                self.metrics[main_key][ambig_type] = {
                    'recall': [],
                    'precision': [],
                    'f1_score': [],
                    'one_found': [],
                    'all_found': []
                }
    
    def update_metrics(self, main_key: str, ambig_type: str, eval_metrics: dict):
        """Update metrics for a specific example"""
        for metric_name, value in eval_metrics.items():
            if metric_name in self.metrics[main_key][ambig_type]:
                self.metrics[main_key][ambig_type][metric_name].append(value)
                self.metrics[main_key]["total"][metric_name].append(value)
    
    def add_zero_metrics(self, main_key: str, ambig_type: str):
        """Add zero values for failed predictions or evaluations"""
        for metric in ['precision', 'recall', 'all_found', 'f1_score', 'one_found']:
            self.metrics[main_key][ambig_type][metric].append(0.0)
            self.metrics[main_key]["total"][metric].append(0.0)
    
    def get_result_metrics(self, eval_metrics: dict) -> dict:
        """Extract metrics for a single result entry"""
        return {
            'precision': eval_metrics.get('precision', 0.0),
            'recall': eval_metrics.get('recall', 0.0),
            'all_found': eval_metrics.get('all_found', 0.0),
            'f1_score': eval_metrics.get('f1_score', 0.0),
            'one_found': eval_metrics.get('one_found', 0.0)
        }
    
    def print_summary(self, short: bool = False):
        """Print summary of all metrics"""
        for main_key in self.metrics:
            print(f"\n{main_key.capitalize()}:")
            if short:
                ambig_type = "total"
                print(f"  {ambig_type.capitalize()}:")
                for metric, values in self.metrics[main_key][ambig_type].items():
                    if values:
                        print(f"    {metric}: {np.mean(values):.4f}")
            else:
                for ambig_type, metrics in self.metrics[main_key].items():
                    print(f"  {ambig_type.capitalize()}:")
                    for metric, values in metrics.items():
                        if values:
                            print(f"    {metric}: {np.mean(values):.4f}")
    
    def get_aggregated_metrics(self) -> dict:
        """Return aggregated metrics instead of raw scores"""
        aggregated = {
            "ambig": {},
            "unambig": {}
        }
        
        for main_key in self.metrics:
            for ambig_type, metrics in self.metrics[main_key].items():
                aggregated[main_key][ambig_type] = {
                    metric: np.mean(values) if values else 0.0
                    for metric, values in metrics.items()
                }
        
        return aggregated