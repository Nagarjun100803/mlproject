#all the code that related to data transformation like one hot encoding, labeling etc..
from dataclasses import dataclass
import pandas as pd
import os 
import sys 
from src.exception import CustomException
from src.logger import logging 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from src.utils import save_obj



@dataclass
class DataTransformationConfig:
    #declaring path for where to store the preprocessor pickle file
    preprocessor_path = os.path.join("artifact", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        #creating all my pickle files responsible for transformation

        '''
        This function is responsible for data transformation
        '''
        try:
            categorical_cols = ['gender', 'race_ethnicity', 
                              'parental_level_of_education', 'lunch', 
                              'test_preparation_course']
            
            numerical_cols = ["reading_score", "writing_score"]

            numerical_pipeline = Pipeline(
                steps = [("imputer",SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler(with_mean=False))]
            )

            categorical_pipeline = Pipeline(
                steps = [("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot_encoder", OneHotEncoder()),
                        ("scaler", StandardScaler(with_mean=False))]
            )

            preprocessor = ColumnTransformer(
                [("numerical_pipeline", numerical_pipeline, numerical_cols),
                 ("categorical_pipeline", categorical_pipeline, categorical_cols)]
            )
            logging.info("Numerical columns scaled sucessfully")
            logging.info("Categorical columns encoded sucessfully")

            return preprocessor # return a column transform object

        except Exception as error:
            raise CustomException(error, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try :
            train_df = pd.read_csv(train_data_path)
            test_df  = pd.read_csv(test_data_path)

            logging.info("Train and Test data readed succesfully")
            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_col = "math_score"
            numerical_cols = ["reading_score", "writing_score"]


            input_feature_train = train_df.drop(target_col, axis=1)
            target_feature_train = train_df[target_col]

            input_feature_test = test_df.drop(target_col, axis=1)
            target_feature_test = test_df[target_col]


            logging.info(
              f"Applying preprocessing object."
            )

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test)

            test_arr  = np.c_[input_feature_test_arr, np.array(target_feature_test)]
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train)]

            logging.info("Converted into numerical representation of a columns")

            save_obj(self.data_transformation_config.preprocessor_path,
                     obj=preprocessor_obj)

            
            return train_arr, test_arr, self.data_transformation_config.preprocessor_path




        except Exception as error:
            raise CustomException(error, sys)



