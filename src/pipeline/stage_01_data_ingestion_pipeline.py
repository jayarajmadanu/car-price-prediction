

from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig


class DataIngestionPipeline:
    def __init__(self, config: DataIngestionConfig):
        self.data_ingestion_config = config
        
    def main(self):
        data_ingestion = DataIngestion(self.data_ingestion_config)
        data_ingestion.downloadDataSet()