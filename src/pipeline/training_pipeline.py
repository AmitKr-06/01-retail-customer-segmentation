import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

import pandas as pd


class TrainingPipeline:
    def run_pipeline(self, raw_data_path: str):
        try:
            logging.info("==== TRAINING PIPELINE STARTED ===")

            # Step 1: Data Ingestion
            ingestion = DataIngestion()
            cleaned_data_path = ingestion.initiate_data_ingestion(
                raw_data_path)

            # Step 2: Data Transformation
            transformation = DataTransformation()
            final_df, scaler_path = transformation.initiate_data_transformation(
                cleaned_data_path)

            # Step 3: Model Training
            trainer = ModelTrainer()
            model_path, silhouette = trainer.initiate_model_training(
                transformation.transformation_config.transformed_data_path,
                n_clusters=4
            )

            # Step 4: Model Evaluation
            evaluator = ModelEvaluation()
            segmented_df = pd.read_csv(
                trainer.model_trainer_config.final_output_path)
            profile = evaluator.profile_clusters(segmented_df)

            feature_cols = [
                'Recency_scaled',
                'Frequency_scaled',
                'Monetary_scaled']
            comparison = evaluator.compare_algorithms(
                segmented_df[feature_cols])

            logging.info("=== TRAINING PIPELINE COMPLETED ===")

            print("\nFinal Silhouette Score (KMeans):", silhouette)
            print("\nCluster Profile:\n", profile)
            print("\nAlgorithm Comparison:\n", comparison)

            return model_path, scaler_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline("data/raw/online_retail.csv")
