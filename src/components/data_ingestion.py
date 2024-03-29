#all the code related to reading the data
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig():
    train_data_path :str = os.path.join("artifact", "train.csv")
    test_data_path  :str = os.path.join("artifact", "test.csv")
    raw_data_path   :str = os.path.join("artifact", "raw.csv")

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered into data ingestion component")
        try:
            df = pd.read_csv("./notebook/data/stud.csv") # we only need to change this logic, from where we read the data
            logging.info("Data readed sucessfully")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path)
            logging.info("Train and Test split is initiated")

            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42
                )
            train_set.to_csv(self.ingestion_config.train_data_path)
            test_set.to_csv(self.ingestion_config.test_data_path)
            logging.info("Data Ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as error:
            logging.info("Error occures during data ingestion")
            raise CustomException(error, sys)
        


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr , path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    mse, score = model_trainer.initiate_model_trainer(train_arr, test_arr)

    print(f"Model score : {score}")
    print(f"Mean squared error : {mse}")








