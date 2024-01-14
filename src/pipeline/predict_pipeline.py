import os 
import sys 
import pandas as pd 
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import load_obj



class PredictPipeline:
    def __init__(self):
        pass 

    def predict(self, features:pd.DataFrame) -> float:
        estimator  = load_obj(os.path.join("artifact", "model.pkl"))
        preprocessor = load_obj(os.path.join("artifact", "preprocessor.pkl"))
        transformed_features = preprocessor.transform(features)
        prediction = estimator.predict(transformed_features)
        return prediction[0]
    



@dataclass
class CustomData :
    gender : str
    race_ethnicity : str
    parental_level_of_education : str
    lunch : str 
    test_preparation_course : str 
    reading_score : int 
    writing_score : int 

    def get_data_as_frame(self) -> pd.DataFrame :
       data = { "gender" : [self.gender],
        "race_ethnicity" : [self.race_ethnicity],
        "parental_level_of_education" : [self.parental_level_of_education],
        "lunch" : [self.lunch],
        "test_preparation_course" : [self.test_preparation_course],
        "reading_score" : [self.reading_score],
        "writing_score" : [self.writing_score]
       }

       return pd.DataFrame(data)
    

if __name__ == "__main__":
    input_ = CustomData('female', 'group C', "bachelor's degree", 'standard', 'completed',  92, 89)
    as_frame = input_.get_data_as_frame()
    pipeline = PredictPipeline()
    print(f"The result is {pipeline.predict(as_frame)}")