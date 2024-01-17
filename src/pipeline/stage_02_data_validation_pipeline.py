from src.logger import logger
from src.entity.config_entity import DataValidationConfig
from src.components.data_validation import DataValidation

class DataValidationPipeline:
    def __init__(self, config: DataValidationConfig):
        self.data_validation_config = config
        
    def main(self):
        data_validation = DataValidation(config= self.data_validation_config)
        validation_status = data_validation.validate_all_columns()
        
        if not validation_status:
            raise Exception("InvalidDatasetError: Passed Invalid Dataset, doesn't contains colunms specified in schema ")
    
    