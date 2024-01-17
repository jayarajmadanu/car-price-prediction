from src.logger import logger
from src.entity.config_entity import DataValidationConfig
import pandas as pd

from src.utils.common import create_directories

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        create_directories([self.config.root_dir])
        
    def validate_all_columns(self) -> bool:
        validation_status = True
        received_dolumns = pd.read_csv(self.config.dataset_file_path).columns
        dataset_schema_columns = self.config.dataset_schema.keys()
        
        for col in dataset_schema_columns:
            if not col in received_dolumns:
                validation_status = False
                logger.info(f"Validation status : {validation_status}, {col} not present in input dataset")
                #break
        with open(self.config.validation_status_file_path, 'w') as f:
            f.write(str(validation_status))
            
        logger.info(f"Dataset Validation Status = {validation_status}")
            
        return validation_status