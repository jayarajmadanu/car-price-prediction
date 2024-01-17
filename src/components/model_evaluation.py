from src.logger import logger
from src.entity.config_entity import ModelEvaluationConfig

import pandas as pd
import mlflow
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class DataEvaluation:
    def __init__(self, config:ModelEvaluationConfig):
        self.config = config
        
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def model_evaluation(self, best_params=None):
        test_data_path = self.config.test_data_path
        model = joblib.load(self.config.model_path)
        
        test_df = pd.read_csv(test_data_path)
        X_test = test_df.iloc[:,0:8]
        y_test = test_df.iloc[:,8]
        
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment('car-price-prediction')
        
        mlflow.autolog()
        with mlflow.start_run():
            predicted_values = model.predict(X_test)
            (rmse, mae, r2) = self.eval_metrics(y_test, predicted_values)
            
            mlflow.log_params(best_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
        logger.info("END")