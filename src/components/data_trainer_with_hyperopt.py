import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from src.logger import logger
from src.entity.config_entity import DataTrainingConfig
from src.utils.common import create_directories

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pandas as pd
import joblib
import os
from hyperopt import hp, fmin, STATUS_OK, Trials, tpe, space_eval
import mlflow


class DataTrainer:
    def __init__(self, config: DataTrainingConfig):
        self.config = config
        create_directories([self.config.root_dir])
        
    def get_or_create_experiment(self,experiment_name):
        if experiment := mlflow.get_experiment_by_name(experiment_name):
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(experiment_name)

    def train(self):
        train_df = pd.read_csv(self.config.train_data_path, )
        logger.info(f"train_df Shape = {train_df.shape}")
        test_df = pd.read_csv(self.config.test_data_path)
        logger.info(f"test_df Shape = {test_df.shape}")
        X_train = train_df.iloc[:,0:8]
        logger.info(f"X_train Shape = {X_train.shape}")
        y_train = train_df.iloc[:,8]
        logger.info(f"y_train Shape = {y_train.shape}")
        X_test = test_df.iloc[:,0:8]
        y_test = test_df.iloc[:,8]
        
        space = hp.choice('Regressor', [
            
            {
                'model': Ridge(),
                'params': {
                    'alpha': hp.choice('alpha', self.config.params['Ridge Regression']['alpha'])
                    },
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            },
            {
                'model': RandomForestRegressor(),
                'params': {
                    'n_estimators': hp.choice('n_estimators', self.config.params['Random Forest']['n_estimators']),
                    'max_features': hp.choice('max_features', self.config.params['Random Forest']['max_features']),
                    'max_depth': hp.choice('max_depth', self.config.params['Random Forest']['max_depth'])
                    },
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        ])
        
        trials = Trials()
        mlflow.set_tracking_uri('https://dagshub.com/jayarajmadanu/car-price-prediction.mlflow')
        mlflow.set_experiment('car-price-prediction')
        exp_id = self.get_or_create_experiment('car-price-prediction')
        with mlflow.start_run(experiment_id=exp_id):
            best_model = fmin(fn=self.objective, space=space, algo=tpe.suggest,max_evals=10, trials=trials )
            best_params = space_eval(space, best_model)
            logger.info("BEST PARAMS ", best_params)
            logger.info("BEST MODEL ", best_model)
        
    
    def evaluate(self,y,pred):
        rmse = np.sqrt(mean_squared_error(y,pred))
        mae = mean_absolute_error(y,pred)
        r2 = r2_score(y,pred)

        return rmse, mae, r2
    
    def objective(self, args):
        model = args['model']
        X = args['X_train']
        y = args['y_train']
        params = args['params']
        y_test = args['y_test']
        x_test = args['X_test']
        
        with mlflow.start_run(nested=True):
            mlflow.sklearn.log_model(model, 'model')
            model.set_params(**params)
            mlflow.log_params(params)
            model.fit(X,y)
            pred = model.predict(x_test)
            rmse,mae,r2 = self.evaluate(y_test,pred)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
                
        return {'loss': rmse, 'status': STATUS_OK}
        