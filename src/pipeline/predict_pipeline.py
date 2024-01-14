import pandas as pd 
from dataclasses import dataclass



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
       data = {"gender" : [self.gender],
        "race_ethnicity" : [self.race_ethnicity],
        "parental_level_of_education" : [self.parental_level_of_education],
        "lunch" : [self.lunch],
        "test_preparation_course" : [self.test_preparation_course],
        "reading_score" : [self.reading_score],
        "writing_score" : [self.writing_score]
       }

       return pd.DataFrame(data)
    
