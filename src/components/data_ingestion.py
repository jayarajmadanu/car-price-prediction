from src.entity.config_entity import DataIngestionConfig
from src.logger import logger
import os
import urllib.request as request
from src.utils.common import create_directories

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def downloadDataSet(self):
        if not os.path.exists(self.config.local_data_file_path):
            filename, headers = request.urlretrieve(
                url=self.config.source_url,
                filename=self.config.local_data_file_path
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info("File already exists")
            
    
    