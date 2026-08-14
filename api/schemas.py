from pydantic import BaseModel, Field


class CustomerRFMInput(BaseModel):
    recency: float = Field(..., gt=0, description="Days since last purchase")
    frequency: float = Field(..., gt=0, description="Number of distinct purchases")
    monetary: float = Field(..., gt=0, description="Total amount spent")



class SegmentPredictionOutput(BaseModel):
    cluster: int
    segment: str
