import sys
import pandas as pd

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from src.exception import CustomException
from src.logger import logging


class ModelEvaluation:
    def __init__(self):
        pass

    def profile_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarizes average RFM values per cluster and labels each segment."""

        try:
            summary = df.groupby('Cluster')[
                ['Recency', 'Frequency', 'Monetary']].mean().round(1)
            summary['Count'] = df['Cluster'].value_counts().sort_index()

            # Simple rule-based naming
            def label_segment(row):
                if row['Recency'] < 30 and row['Frequency'] > 8:
                    return 'Champions'
                elif row['Recency'] < 30:
                    return 'New/Promising'
                elif row['Recency'] < 100:
                    return 'Loyal'
                else:
                    return 'At Risk'

            summary['Segment'] = summary.apply(label_segment, axis=1)
            logging.info(f"Cluster profile:\n{summary}")

            return summary

        except Exception as e:
            raise CustomException(e, sys)

    def compare_algorithms(self, X, n_clusters: int = 4):
        """Compares KMeans against DBSCAN and Hierarchical clustering."""

        try:
            results = {}

            # Hierarchical
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            hier_labels = hierarchical.fit_predict(X)
            results['Hierarchical'] = silhouette_score(X, hier_labels)

            # DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(X)
            n_dbscan_clusters = len(set(dbscan_labels)) - \
                (1 if -1 in dbscan_labels else 0)

            if n_dbscan_clusters > 1:
                results['DBSCAN'] = silhouette_score(X, dbscan_labels)
            else:
                results['DBSCAN'] = None

            logging.info(f"Algorithm comparison: {results}")
            return results

        except Exception as e:
            raise CustomException(e, sys)


# Quick test
if __name__ == "__main__":
    df = pd.read_csv("artifacts/customer_segments.csv")

    evaluator = ModelEvaluation()

    profile = evaluator.profile_clusters(df)
    print("Cluster Profile:\n", profile)

    feature_cols = ['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']
    comparison = evaluator.compare_algorithms(df[feature_cols])
    print("\nAlgorithm Comparison (Silhouette Scores):\n", comparison)
