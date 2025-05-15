from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
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
    logo: str 
    company_name: str
    job_location: str
    job_min_salary: float
    job_max_salary: float
    reason: str

# CV Analysis Models
class CVAnalysisRequest(BaseModel):
    file_path: str
    
class CVAnalysisResponse(BaseModel):
    success: bool
    resume_id: Optional[int] = None
    skills: Optional[List[str]] = []
    education: Optional[List[Dict[str, Any]]] = []
    experience: Optional[List[Dict[str, Any]]] = []
    summary: Optional[str] = ""
    industry: Optional[str] = ""
    seniority: Optional[str] = ""
    language: Optional[str] = "en"
    error: Optional[str] = None
    
# CV Recommendation Models
class CVRecommendationRequest(BaseModel):
    resume_id: int
    limit: Optional[int] = 10
    
class CVJobMatchResponse(BaseModel):
    job_id: int
    job_title: str
    match_score: float
    skill_match_score: Optional[float] = None
    content_similarity: Optional[float] = None
    matched_skills: List[str]
    missing_skills: List[str]
    company_name: str
    logo: str
    employer_id: int
    category_id: int
    level_id: int
    job_location: str
    job_min_salary: float
    job_max_salary: float
    reason: str

class ResumeJobMatchResponse(BaseModel):
    resume_id: int
    student_name: Optional[str] = None
    match_score: float
    skill_match_score: Optional[float] = None
    content_similarity: Optional[float] = None
    matched_skills: List[str]
    missing_skills: List[str]
    resume_title: str
    reason: str
    
class CVRecommendationResponse(BaseModel):
    recommendations: List[CVJobMatchResponse]

class ResumeRecommendationResponse(BaseModel):
    recommendations: List[ResumeJobMatchResponse]
