import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model_path = "artifacts/kmeans_model.pkl"
        self.scaler_path = "artifacts/scaler.pkl"
        self.segments_data_path = "artifacts/customer_segments.csv"

    def _build_segment_map(self) -> dict:
        """Dynamically determines which cluster number corresponds to which segment,
        based on the actual trained model's cluster centers (not hardcoded numbers)."""
        df = pd.read_csv(self.segments_data_path)
        profile = df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

        # Rank clusters: low Recency + high Frequency + high Monetary = better segment
        profile['score'] = (
            profile['Frequency'].rank() +
            profile['Monetary'].rank() -
            profile['Recency'].rank()
        )

        sorted_clusters = profile.sort_values('score', ascending=False).index.tolist()

        labels = ["Champions", "Loyal", "New/Promising", "At Risk"]
        segment_map = {cluster: labels[i] for i, cluster in enumerate(sorted_clusters)}
        return segment_map

    def predict(self, recency: float, frequency: float, monetary: float) -> dict:
        try:
            model = load_object(self.model_path)
            scaler = load_object(self.scaler_path)
            segment_map = self._build_segment_map()

            recency_log = np.log1p(recency)
            frequency_log = np.log1p(frequency)
            monetary_log = np.log1p(monetary)

            input_df = pd.DataFrame([{
                'Recency_log': recency_log,
                'Frequency_log': frequency_log,
                'Monetary_log': monetary_log
            }])

            scaled_input = scaler.transform(input_df)
            scaled_input_df = pd.DataFrame(
                scaled_input, columns=['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']
            )

            cluster = model.predict(scaled_input_df)[0]
            segment_name = segment_map.get(int(cluster), "Unknown")

            logging.info(f"Prediction made: Cluster={cluster}, Segment={segment_name}")

            return {
                "cluster": int(cluster),
                "segment": segment_name
            }

        except Exception as e:
            raise CustomException(e, sys)