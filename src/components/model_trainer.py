import os
import sys
import pandas as pd


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dataclasses import dataclass


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join('artifacts', 'kmeans_model.pkl')
    final_output_path: str = os.path.join('artifacts', 'customer_segments.csv')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def find_optimal_k(self, X, k_range = range(2, 11)):
        """Runs elbow method + silhouette score across a range of K values."""

        try:
            inertia = []
            silhouette_scores = []

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                inertia.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(X, labels))

            logging.info(f"Inertia values: {inertia}")
            logging.info(f"Silhouette scores: {silhouette_scores}")

            return inertia, silhouette_scores

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_model_training(self, transformed_data_path: str, n_clusters: int = 4):
        try:
            df = pd.read_csv(transformed_data_path)
            logging.info("Loaded transformed RFM data for model training")


            feature_cols = ['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']
            X = df[feature_cols]



            # run elbow/silhouette check (mainly for experimentation/logs)
            self.find_optimal_k(X)


            # Train model with chosen K
            kmeans_final  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['Cluster'] = kmeans_final.fit_predict(X)

            final_score = silhouette_score(X, df['Cluster'])
            logging.info(f"Final KMeans (K={n_clusters}) silhouette score: {final_score}")


            # Save model
            save_object(self.model_trainer_config.trained_model_path, kmeans_final)


            # Save final segmented customer data
            os.makedirs(os.path.dirname(self.model_trainer_config.final_output_path), exist_ok=True)
            df.to_csv(self.model_trainer_config.final_output_path, index=False)

            logging.info("Model training completed successfully")

            return self.model_trainer_config.trained_model_path, final_score


        except Exception as e:
            raise CustomException(e, sys)

