import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.modle_path = "artifacts/kmeans_model.pkl"
        self.scaler_path = "artifacts/scaler.pkl"


    def predict(self, recency: float, frequency: float, monetary: float) -> dict:
        try:
            model = load_object(self.modle_path)
            scaler = load_object(self.scaler_path)



            # Apply same log tranform as training
            recency_log = np.log1p(recency)
            frequency_log = np.log1p(frequency)
            monetary_log = np.log1p(monetary)

            input_df = pd.DataFrame([{
                'Recency_log': recency_log,
                'Frequency_log': frequency,
                'Monetary_log': monetary
            }])



            # Apply same scaler as training
            scaled_input = scaler.transform(input_df)
            scaled_input_df = pd.DataFrame(scaled_input, columns=['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled'])

            cluster = model.predict(scaled_input_df)[0]

            segment_map ={
                0: "New/Promising",
                1: "Loyal",
                2: "At Risk",
                3: "Champions"
            }



            segment_name = segment_map.get(cluster, "Unknown")

            logging.info(f"Prediction made: Cluster={cluster}, Segment={segment_name}")


            return{
                "cluster": int(cluster),
                "segment": segment_name
            }



        except Exception as e:
            raise CustomException(e, sys)



