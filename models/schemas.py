from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class JobBase(BaseModel):
    job_title: str
    description: str
    company: Optional[str] = None
    location: Optional[str] = None
    salary_range: Optional[str] = None

class JobResponse(JobBase):
    job_id: int

    class Config:
        orm_mode = True

class RatingRequest(BaseModel):
    user_id: int
    job_id: int
    rating: float

class ActivityLogRequest(BaseModel):
    user_id: int
    job_id: int
    action_type: str

class RecommendationRequest(BaseModel):
    user_id: int
    top_n: Optional[int] = 5

class RecommendationResponse(BaseModel):
    job_id: int
    score: float
    job_title: str
    job_description: str
    job_requirements: str
    job_min_salary: float
    job_max_salary: float
