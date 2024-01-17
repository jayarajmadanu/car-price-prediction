from sklearn.base import BaseEstimator, TransformerMixin
from src.entity.config_entity import DataTransformationConfig
from src.logger import logger
from src.utils.common import create_directories, save_object

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        create_directories([self.config.root_dir])
        
    def summarise_df(self,df, dataset_summary_path, title:str = ""):
        with open(dataset_summary_path, 'a') as f:
            f.write(title + '\n')
            f.write("df.info() \n")
            f.write(df.info())
            f.write("df.describe(include='number') \n")
            f.write(df.describe(include='number'))
            f.write("df.describe(include='object') \n")
            f.write(df.describe(include='object'))
            
    def transform_data(self) -> ColumnTransformer:
        try:
            
            ## NOTE: Colunm Transformer will change the order of colunms after applying transformation, so check the value of index mentioned in colunmTransformer after eact CT
            
            tr1 = ColumnTransformer([
                ('year_to_age_converter', YearToAgeConverter(),[0])
            ], remainder='passthrough')
            tr2 = ColumnTransformer([
                ('imputer numeric', SimpleImputer(strategy="median"), [0,1]),
                ("imputer categorical",SimpleImputer(strategy="most_frequent"), [2,3,4,5])
            ], remainder='passthrough')
            tr3 = ColumnTransformer([
                #("imputer",SimpleImputer(strategy="most_frequent"), [2,3,4,5])
            ], remainder='passthrough')
            tr4 = ColumnTransformer([
                ('ordinal_encoder', OrdinalEncoder(categories=[['Test Drive Car', 'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']]),[5])
            ], remainder='passthrough')
            tr5 = ColumnTransformer([
                ('custom_onehotencoder', CustomOneHotEncoder(colunms=['fuel', 'seller_type', 'transmission']),[3,4,5])
            ], remainder='passthrough')
            tr6 = ColumnTransformer([
                ("scalar",StandardScaler(with_mean=False), slice(0,8))
            ], remainder='passthrough')
            
            pipeline = Pipeline(
                steps=[
                    ('tr1', tr1),
                    ('tr2', tr2),
                    #('tr3', tr3),
                    ('tr4', tr4),
                    ('tr5', tr5),
                    ('tr6', tr6),
                ]
            )
            return pipeline
        except Exception as e:
            logger.info(e)
            
    def initiate_data_transformation(self):
        dataset_file_path = self.config.dataset_file_path
        df = pd.read_csv(dataset_file_path)
        #dataset_summary_path = self.config.dataset_summary_path
        #self.summarise_df(df,dataset_summary_path, "Before Data Transformation")
        
        # Drop 'name' colunm
        df.drop('name', axis=1, inplace=True)
        
        # Drop Outliers
        df.drop(df[ (df['km_driven'] > 400000) | (df['selling_price'] > 8000000 )].index, axis=0, inplace=True)
        
        X = df.drop(self.config.targer_colunm, axis=1)
        y = df[self.config.targer_colunm]
        X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size=self.config.test_size, random_state=self.config.random_state)
        
        logger.info(f'colunm names = {X.columns} and shape is {X_train.shape}')
        preprocessor = self.transform_data()
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        train_dataset = np.c_[X_train_processed, np.array(y_train)]
        test_dataset = np.c_[X_test_processed, np.array(y_test)]
        
        train_dataset = pd.DataFrame(train_dataset)
        logger.info(f"Created train dataset at location {self.config.train_dataset_file_path} with shape {train_dataset.shape}")
        train_dataset.to_csv(self.config.train_dataset_file_path, index=False)
        test_dataset = pd.DataFrame(test_dataset)
        logger.info(f"Created test dataset at location {self.config.test_dataset_file_path} with shape {test_dataset.shape}")
        test_dataset.to_csv(self.config.test_dataset_file_path, index=False)
        
        save_object(self.config.preprocessor_obj_path, preprocessor)
        
        #self.summarise_df(df,dataset_summary_path, "After Data Transformation")
        
        


        
class YearToAgeConverter(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        self.max_year= X['year'].max()
        return self
    
    def transform(self, X):
        X.insert(0,'age', self.max_year - X['year'] + 1)
        X.drop('year', axis=1, inplace=True)
        return X
    
class CustomOneHotEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, colunms:list=None):
        self.colunms = colunms
        #self.encoder = OneHotEncoder(handle_unknown='ignore')
    
    def fit(self, X, y=None):
        #for col in self.columns:
        #    self.unique_values_per_column[col] = X[col].unique()
        #return self
        #self.encoder= self.encoder.fit(X,y)
        return self
    def transform(self, X):
        '''X=self.encoder.transform(X)
        logger.info(f"colunms created in oneHotEncoder are {self.encoder.get_feature_names_out()}")
        encoded_df = pd.DataFrame(X, columns=self.encoder.get_feature_names_out(self.columns))
        encoded_df.drop(['x0_CNG','x0_Electric', 'x0_LPG','x1_Trustmark Dealer','x2_Automatic'], axis=1, inplace=True)
        return encoded_df'''
        logger.info(f"Shape of X in OneHotEncoder is {X.shape}")
        encoded_df = pd.get_dummies(data=pd.DataFrame(X,columns=self.colunms), columns=self.colunms, dtype=int)
        logger.info(f"colunms created in oneHotEncoder are {encoded_df.columns}")
        for col in ['fuel_CNG','fuel_LPG','fuel_Electric','seller_type_Trustmark Dealer','transmission_Automatic']:
            if col in encoded_df.columns:
                encoded_df.drop(col, axis=1, inplace=True)
        return encoded_df
        
    