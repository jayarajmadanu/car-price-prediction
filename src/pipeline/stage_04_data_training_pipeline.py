from src.logger import logger
from src.entity.config_entity import DataTrainingConfig
from src.components.data_trainer import DataTrainer

class DataTrainingPipeline:
    def __init__(self, config: DataTrainingConfig):
        self.data_training_config = config
        
    def main(self):
        data_trainer = DataTrainer(config= self.data_training_config)
        best_params = data_trainer.train()
        return best_params
        
        
    