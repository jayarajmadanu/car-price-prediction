from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from src.logger import logger
from src.entity.config_entity import DataTrainingConfig
from src.utils.common import create_directories, save_object

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pandas as pd
from xgboost import XGBRegressor
import os

class DataTrainer:
    def __init__(self, config: DataTrainingConfig):
        self.config = config
        create_directories([self.config.root_dir])
        
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
        
        models = {
            #"Linear Regression": LinearRegression(),
            #"Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            #"Lasso": Lasso(),
            "SVR":SVR(),
            "Elastic Net": ElasticNet(max_iter=1000000),
            "XGBRegressor": XGBRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            #"Ridge Regression" : Ridge()
        }
        logger.info("TRAINING MODELS")
        report = self.evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=self.config.params)
        
        best_model_name = ""
        max_r2_score = 0
        for model_name in report.keys():
            if max_r2_score < report[model_name]['R2_score_test']:
                max_r2_score = report[model_name]['R2_score_test']
                best_model_name = model_name
        
        best_model = report[best_model_name]['model']
        logger.info(f"Best model found is {best_model_name} with R2 Score of {report[best_model_name]} and best parameters are {report[best_model_name]['best_params']}")
        #self.config.model_name = best_model_name + '.pkl'
        #joblib.dump(best_model, os.path.join(self.config.root_dir, self.config.model_name))
        save_object(os.path.join(self.config.root_dir, self.config.model_name), best_model)
        return report[best_model_name]['best_params']
        
    def evaluate_models(self,X_train,y_train, X_test,y_test, models:dict, params:dict):
        model_keys = models.keys()
        report = {}
        
        for model_name in model_keys:
            model = models[model_name]
            parameters = params[model_name]

            # GridSearchCV will get best hypermaters for each model
            gs = GridSearchCV(estimator=model, param_grid=parameters, cv=5, refit=True)
            gs.fit(X_train, y_train)

            # now test the model with training data

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            y_train_pred = model.predict(X_train)
            train_model_score = r2_score(y_train, y_train_pred)
            report[model_name] = {
                'model' : model,
                'R2_score_test' : test_model_score,
                'R2_score_train' : train_model_score,
                'best_params': gs.best_params_
            }
        logger.info(f'Model Evaluation report: \n{report}')
        return report