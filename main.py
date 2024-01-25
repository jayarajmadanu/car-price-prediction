from src.config.configuration import ConfigurationManager
from src.logger import logger
from src.pipeline.stage_01_data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.stage_02_data_validation_pipeline import DataValidationPipeline
from src.pipeline.stage_03_data_transformation_ppeline import DataTransformationPipeline
from src.pipeline.stage_04_data_training_pipeline import DataTrainingPipeline
from src.pipeline.stage_05_data_evaluation_pipeline import ModelEvaluationPipeline

config = ConfigurationManager()

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion_pipeline = DataIngestionPipeline(data_ingestion_config)
    data_ingestion_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_validation_config = config.get_data_validation_config()
    data_validation_pipeline = DataValidationPipeline(data_validation_config)
    data_validation_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_transformation_config = config.get_data_transformation_config()
    data_transformation_pipeline = DataTransformationPipeline(data_transformation_config)
    data_transformation_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Training stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_training_config = config.get_data_training_config()
    data_training_pipeline = DataTrainingPipeline(data_training_config)
    model_training_best_params = data_training_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
    

STAGE_NAME = "Model evaluation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_evaluation_config = config.get_model_evaluation_config()
   model_evaluation = ModelEvaluationPipeline(config=model_evaluation_config)
   model_evaluation.main(best_params=model_training_best_params)
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
