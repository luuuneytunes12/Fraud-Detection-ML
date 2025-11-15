from pydantic import BaseModel, field_validator
from typing import List

class PredictionInput(BaseModel): 
    data = List[float]
    # Validator to check if the input list contains 30 values
    @field_validator("data")
    @classmethod
    def check_length(cls, v):
        if len(v) != 30:
            raise ValueError("data must contain exactly 30 float values")
        return v
 
    # Provide an example schema for documentation
    class Config:
        json_schema_extra = {
            "example": {
                "data": [5.1, 3.5, 1.4, 0.2], 
            }
        }
    