from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from models.database import get_db
from models.schemas import RecommendationRequest, RecommendationResponse
from services.recommender_service import RecommenderManager

router = APIRouter()

@router.post("/recommend", response_model=List[RecommendationResponse])
async def get_recommendations(
    request: RecommendationRequest, 
    db: Session = Depends(get_db)
):
    try:
        # Unpack the recommender and jobs dataframe
        recommender, jobs_df = RecommenderManager.get_recommender(db)
        
        # Get recommendations
        recommendations = recommender.recommend(
            user_id=request.user_id,
            top_n=request.top_n
        )
        
        # Merge recommendation scores with job details
        detailed_recommendations = []
        for rec in recommendations:
            job_details = RecommenderManager.get_job_details(rec['job_id'], jobs_df)
            if job_details:
                detailed_rec = {
                    **rec,
                    "job_title": job_details['job_title'],
                    "job_description": job_details['job_description'],
                    "job_requirements": job_details['job_requirements'],
                    "job_min_salary": job_details['job_min_salary'],
                    "job_max_salary": job_details['job_max_salary'],
                }
                detailed_recommendations.append(detailed_rec)
        
        return detailed_recommendations
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))