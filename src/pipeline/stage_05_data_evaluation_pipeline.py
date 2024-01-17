from src.logger import logger
from src.entity.config_entity import ModelEvaluationConfig
from src.components.model_evaluation import DataEvaluation

class ModelEvaluationPipeline:
    def __init__(self, config:ModelEvaluationConfig):
        self.config = config
        
    def main(self, best_params=None):
        model_evaluation = DataEvaluation(config = self.config)
        model_evaluation.model_evaluation(best_params=best_params)
        