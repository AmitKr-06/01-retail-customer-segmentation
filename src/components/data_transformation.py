import os
import sys
import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass 
class DataTransformationConfig:
    scaler_obj_path: str = os.path.join('artifacts', 'scaler.pkl')
    transformed_data_path: str = os.path.join('artifacts','rfm_transformed.csv')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()



    def build_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Builds RFM table from cleaned transaction-level data."""

        try:
            df['TotalPrice'] = df['Quantity'] * df['Price']
            reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

            rfm = df.groupby('Customer ID').agg({
                'InvoiceDate': lambda x: (reference_date - x.max()).days,
                'Invoice': 'nunique',
                'TotalPrice': 'sum'
            }).reset_index()


            rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
            logging.info(f"RFM table build with shape: {rfm.shape}")
            return rfm

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, cleaned_data_path: str):
        try:
            df = pd.read_csv(cleaned_data_path, parse_dates=['InvoiceDate'])
            logging.info("Loaded cleaned data for transformation")

            rfm = self.build_rfm(df)


            # Log transform to reduce skew
            rfm['Recency_log'] = np.log1p(rfm['Recency'])
            rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
            rfm['Monetary_log'] = np.log1p(rfm['Monetary'])


            # Scale
            features = rfm[['Recency_log', 'Frequency_log', 'Monetary_log']]
            scaler = StandardScaler()
            scaled_array = scaler.fit_transform(features)

            rfm_scaled = pd.DataFrame(
                scaled_array,
                columns=['Recency_scaled','Frequency_scaled','Monetary_scaled']

            )


            final_df = pd.concat([rfm, rfm_scaled], axis=1)


            os.makedirs(os.path.dirname(self.transformation_config.transformed_data_path), exist_ok=True)

            final_df.to_csv(self.transformation_config.transformed_data_path, index=False)


            # Save the scaler for later use in prediction
            save_object(self.transformation_config.scaler_obj_path, scaler)

            logging.info("Data transformation completed successfully")

            return final_df, self.transformation_config.scaler_obj_path

        except Exception as e:
            raise CustomException(e, sys)





