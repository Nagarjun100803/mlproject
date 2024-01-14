import os 
import sys 
from src.exception import CustomException
from src.logger import logging
import pickle 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score




def save_obj(file_path, obj):
    try :
       dir_path = os.path.dirname(file_path)
       os.makedirs(dir_path, exist_ok=True)
       
       with open(file_path, "wb") as file_obj:
           pickle.dump(obj,file_obj)
           
    except Exception as error:
        raise CustomException(error, sys)
    

def evaluate_models(models:dict, params:dict, X_train, X_test, y_train, y_test):
    try : 
        reports = {}
        for est_name, est in models.items():
            model = est
            param_grid = params[est_name]

            grid_search = GridSearchCV(
                estimator=est,
                param_grid=param_grid,
                scoring="neg_mean_squared_error",
                cv = 5
            )
            grid_search.fit(X_train, y_train)
            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            r2_score_train_data = r2_score(y_train, y_pred_train)
            r2_score_test_data = r2_score(y_test, y_pred_test)

            reports[est_name] = r2_score_test_data

        return reports
    
    except Exception as error:
        raise CustomException(error, sys)
    


def load_obj(file_path):
    try :
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except Exception as error:
        raise CustomException(error, sys)
    

