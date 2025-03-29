import pandas as pd
from functools import lru_cache
from models.models import Job, Rating, ActivityLog
from sqlalchemy.orm import Session
from models.hybrid_model import create_hybrid_recommender

class RecommenderManager:
    @staticmethod
    @lru_cache(maxsize=1)
    def get_recommender(db: Session):
        # Fetch jobs from database
        jobs = db.query(Job).all()
        jobs_data = [
            {
                "job_id": job.job_id,
                "job_title": job.job_title,
                "job_description": job.job_description,
                "job_requirements": job.job_requirements,
                "job_location": job.job_location,
                "job_min_salary": job.job_min_salary,
                "job_max_salary": job.job_max_salary
            } for job in jobs
        ]
        
        # Fetch ratings from database
        ratings = db.query(Rating).all()
        ratings_data = [
            {
                "user_id": rating.student_id,
                "job_id": rating.job_id
            } for rating in ratings
        ]
        
        # Fetch activity logs from database
        activity_logs = db.query(ActivityLog).all()
        activity_logs_data = [
            {
                "user_id": log.user_id,
                "job_id": log.job_id,
                "action_type": log.action_type,
                "timestamp": log.timestamp.isoformat()
            } for log in activity_logs
        ]
        
        # Convert to DataFrames
        jobs_df = pd.DataFrame(jobs_data)
        ratings_df = pd.DataFrame(ratings_data)
        activity_logs_df = pd.DataFrame(activity_logs_data)
                
        # Create and cache recommender
        recommender = create_hybrid_recommender(
            jobs_df, 
            ratings_df, 
            activity_logs_df
        )
        
        return recommender, jobs_df

    @staticmethod
    def get_job_details(job_id, jobs_df):
        job = jobs_df[jobs_df['job_id'] == job_id].to_dict('records')
        return job[0] if job else None

    @classmethod
    def invalidate_cache(cls):
        cls.get_recommender.cache_clear()