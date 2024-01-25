

from src.components.data_ingestion import DataIngestion
from src.config.configuration import ConfigurationManager
from src.entity.config_entity import DataIngestionConfig


class DataIngestionPipeline:
    def __init__(self, config: DataIngestionConfig = None):
        if config == None:
            config = ConfigurationManager().get_data_ingestion_config()
        self.data_ingestion_config = config
        
    def main(self):
        data_ingestion = DataIngestion(self.data_ingestion_config)
        data_ingestion.downloadDataSet()
        
        
if __name__ == '__main__':
    try:
        obj = DataIngestionPipeline()
        obj.main()
    except Exception as e:
        raise e