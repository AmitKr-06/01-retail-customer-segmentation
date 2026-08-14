from fastapi import FastAPI, HTTPException
from api.schemas import CustomerRFMInput, SegmentPredictionOutput


from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging


app = FastAPI(
    title="Customer Segmentation API",
    description="Predicts customer segment (Champions, Loyal, At Risk, New/Promising) based on RFM values",
    version="1.0.0")


@app.get("/")
def read_root():
    return {"message": "Customer Segmentation API is running"}


@app.post("/predict", response_model=SegmentPredictionOutput)
def predict_segment(data: CustomerRFMInput):
    try:
        pipeline = PredictPipeline()
        result = pipeline.predict(
            recency=data.recency,
            frequency=data.frequency,
            monetary=data.monetary
        )

        return result

    except Exception as e:
        logging.info(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please check input values.")
