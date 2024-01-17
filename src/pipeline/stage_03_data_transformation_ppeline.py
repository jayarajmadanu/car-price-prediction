from src.logger import logger
from src.entity.config_entity import DataTransformationConfig
from src.utils.common import create_directories
from src.components.data_transformation import DataTransformation

class DataTransformationPipeline:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        
    def main(self):
        data_transformation = DataTransformation(config= self.config)
        data_transformation.initiate_data_transformation()