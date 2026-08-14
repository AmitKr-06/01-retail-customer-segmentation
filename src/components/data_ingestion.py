import sys
import os
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    cleaned_data_path: str = os.path.join('artifacts','cleaned.csv')



class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self, source_path: str):
        logging.info("Starting data ingestion")
        try:
            df = pd.read_csv(source_path)
            logging.info(f"Read raw dataset with shape {df.shape}")


            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)



            # ---- Cleaning steps (notebook)
            df = df.dropna(subset=['Customer ID'])
            df['Customer ID'] = df['Customer ID'].astype(int)
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            df['Invoice'] = df['Invoice'].astype(str)


            # Remove Cancellations
            df = df[~df['Invoice'].str.startswith('C')]

            # Remove duplicates
            df = df.drop_duplicates()

            # Remove non-positive Quantity/Price
            df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]

            # Remove known non-product StockCodes
            non_product_codes = ['POST','ADJUST','ADJUST2','M','TEST001','TEST002','D','BANK CHARGES','C2']

            df = df[~df['StockCode'].isin(non_product_codes)]

            df = df.reset_index(drop=True)
            logging.info(f"Cleaned dataset shape:  {df.shape}")

            df.to_csv(self.ingestion_config.cleaned_data_path, index=False)
            logging.info("Data ingestion completed successfully")

            return self.ingestion_config.cleaned_data_path

        except Exception as e:
            raise CustomException(e, sys)


     
    
