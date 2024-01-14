#Model Training
import os 
import numpy as np 
import pandas as pd 
import sys 
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from dataclasses import dataclass
from sklearn.neighbors import KNeighborsRegressor
from src.utils import evaluate_models, save_obj
from sklearn.metrics import r2_score, mean_squared_error



@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifact", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try :
            logging.info("Model training is started")
            X_train, X_test, y_train, y_test = (
                train_arr[:,:-1], test_arr[:,:-1],
                train_arr[:,-1], test_arr[:,-1]
                )

            models = {
                "Linear Regression" : LinearRegression(),
                "Ridge" : Ridge(),
                "Lasso" : Lasso(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "Ada Boosting"  : AdaBoostRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor() 
            }

            params = {
                "Decision Tree" : {
                    "criterion" : ["squared_error", "absolute_error"]
                },
                "Random Forest" : {
                    # "criterion" : ["squared_error", "absolute_error"]
                    "n_estimators" : [8, 16, 32, 64, 128]
                },
                "Gradient Boosting" : {
                    "learning_rate" : [.0001, .001, .01, 1, 10],
                    "n_estimators" : [8, 16, 32, 64, 128]
                },
                "Linear Regression" : {},
                "Ridge" : {
                    "alpha" : [.0001, .001, .01, 1, 10],
                },
                "Lasso" : {
                    "alpha" : [.0001, .001, .01, 1, 10],
                },
                "Ada Boosting" : {
                    "learning_rate" : [.1, .01, .5, .0001],
                    "n_estimators" : [8, 16, 32, 64, 128]
                }
            }
            reports : dict = evaluate_models(
                models, params,X_train,X_test,y_train, y_test
            )

            best_score = max(list(reports.values()))
            best_model_index = list(reports.values()).index(best_score)
            best_model_name = list(models)[best_model_index]

            best_model = models[best_model_name]
            print(best_model)
            print("=="*15)
            print(best_model.get_params())

            if best_score <= 0.5 :
                raise CustomException("No best models is found")
            logging.info("Best model found for both train and test data")

            save_obj(
                file_path= self.model_trainer_config.trained_model_path,
                obj = best_model
            )     
            predicted = best_model.predict(X_test)
            score, mse = r2_score(y_test, predicted), mean_squared_error(y_test, predicted)

            return mse, score

    
        except Exception as error :
            raise CustomException(error, sys)



