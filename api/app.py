# import pandas as pd
# import numpy as np
# from fastapi import FastAPI, HTTPException, Depends
# from pydantic import BaseModel
# from typing import List, Optional

# # Nhúng các model recommendation
# from datetime import datetime, timezone, timedelta
# from models.hybrid_model import create_hybrid_recommender
# app = FastAPI(title="Job Recommendation Microservice")

# # Mô hình dữ liệu
# class JobModel(BaseModel):
#     job_id: int
#     title: str
#     description: str
#     company: Optional[str] = None
#     location: Optional[str] = None
#     salary_range: Optional[str] = None

# class RatingModel(BaseModel):
#     user_id: int
#     job_id: int
#     rating: float

# class ActivityLogModel(BaseModel):
#     user_id: int
#     job_id: int
#     action_type: str
#     timestamp: str

# class RecommendationRequest(BaseModel):
#     user_id: int
#     top_n: Optional[int] = 5

# class RecommendationResponse(BaseModel):
#     job_id: int
#     score: float
#     title: str
#     description: str
#     company: Optional[str] = None
#     location: Optional[str] = None
#     salary_range: Optional[str] = None

# # Khởi tạo recommender global 
# class RecommenderManager:
#     _instance = None
    
#     @classmethod
#     def get_recommender(cls):
#         if cls._instance is None:
#             # Khởi tạo recommender với dữ liệu mẫu
#             jobs = [
#                 {
#                     "job_id": 1, 
#                     "title": "Senior Python Developer", 
#                     "description": "Senior Python developer with Django and Flask experience. Create scalable backend solutions.", 
#                     "company": "Tech Innovations Inc.",
#                     "location": "San Francisco, CA",
#                     "salary_range": "$90,000 - $120,000",
#                     "skills": ["Python", "Django", "Flask", "REST API", "Microservices"]
#                 },
#                 {
#                     "job_id": 2, 
#                     "title": "Java Backend Engineer", 
#                     "description": "Experienced Java developer for enterprise solutions. Work on complex distributed systems.", 
#                     "company": "Enterprise Solutions LLC",
#                     "location": "New York, NY", 
#                     "salary_range": "$95,000 - $125,000",
#                     "skills": ["Java", "Spring Boot", "Microservices", "Kafka", "Docker"]
#                 },
#                 {
#                     "job_id": 3, 
#                     "title": "Frontend React Developer", 
#                     "description": "React and TypeScript expert for modern web applications. Build responsive and interactive UIs.", 
#                     "company": "Digital Trends Corp",
#                     "location": "Remote",
#                     "salary_range": "$85,000 - $110,000",
#                     "skills": ["React", "TypeScript", "Redux", "GraphQL", "Responsive Design"]
#                 },
#                 {
#                     "job_id": 4, 
#                     "title": "Data Scientist", 
#                     "description": "Machine learning specialist with advanced analytics skills. Develop predictive models and AI solutions.", 
#                     "company": "AI Research Labs",
#                     "location": "Boston, MA",
#                     "salary_range": "$100,000 - $140,000",
#                     "skills": ["Python", "Machine Learning", "TensorFlow", "SQL", "Data Analysis"]
#                 },
#                 {
#                     "job_id": 5, 
#                     "title": "DevOps Engineer", 
#                     "description": "Cloud infrastructure and CI/CD expert. Optimize deployment and scalability.", 
#                     "company": "Cloud Dynamics",
#                     "location": "Seattle, WA",
#                     "salary_range": "$95,000 - $130,000",
#                     "skills": ["AWS", "Kubernetes", "Docker", "Jenkins", "Terraform"]
#                 },
#                 {
#                     "job_id": 6, 
#                     "title": "Full Stack Developer", 
#                     "description": "Versatile developer proficient in both frontend and backend technologies.", 
#                     "company": "Innovative Solutions Tech",
#                     "location": "Austin, TX",
#                     "salary_range": "$88,000 - $115,000",
#                     "skills": ["JavaScript", "Node.js", "React", "MongoDB", "Express"]
#                 },
#                 {
#                     "job_id": 7, 
#                     "title": "Machine Learning Engineer", 
#                     "description": "Advanced ML engineer to develop cutting-edge AI algorithms and solutions.", 
#                     "company": "Cognitive Systems",
#                     "location": "San Jose, CA",
#                     "salary_range": "$110,000 - $150,000",
#                     "skills": ["Python", "TensorFlow", "PyTorch", "Deep Learning", "AI"]
#                 },
#                 {
#                     "job_id": 8, 
#                     "title": "Cloud Solutions Architect", 
#                     "description": "Design and implement scalable cloud infrastructure solutions.", 
#                     "company": "CloudNative Systems",
#                     "location": "Chicago, IL",
#                     "salary_range": "$120,000 - $160,000",
#                     "skills": ["AWS", "Azure", "Cloud Architecture", "Kubernetes", "Microservices"]
#                 }
#             ]
            
#             ratings = [
#                 {"user_id": 1, "job_id": 1, "rating": 4.5},  # Python Developer
#                 {"user_id": 1, "job_id": 2, "rating": 3.8},  # Java Backend
#                 {"user_id": 1, "job_id": 4, "rating": 4.2},  # Data Scientist
#                 {"user_id": 2, "job_id": 3, "rating": 4.0},  # Frontend React
#                 {"user_id": 2, "job_id": 6, "rating": 4.3},  # Full Stack
#                 {"user_id": 3, "job_id": 5, "rating": 3.9},  # DevOps
#                 {"user_id": 3, "job_id": 8, "rating": 4.1}   # Cloud Solutions
#             ]
#             base_time = datetime.now(timezone.utc)
#             activity_logs = [
#                 # User 1 interactions
#                 {
#                     "user_id": 1, 
#                     "job_id": 1, 
#                     "action_type": "VIEW", 
#                     "timestamp": base_time.isoformat()
#                 },
#                 {
#                     "user_id": 1, 
#                     "job_id": 4, 
#                     "action_type": "SAVE", 
#                     "timestamp": (base_time - timedelta(hours=2)).isoformat()
#                 },
#                 {
#                     "user_id": 1, 
#                     "job_id": 2, 
#                     "action_type": "APPLY", 
#                     "timestamp": (base_time - timedelta(days=1)).isoformat()
#                 },
                
#                 # User 2 interactions
#                 {
#                     "user_id": 2, 
#                     "job_id": 3, 
#                     "action_type": "VIEW", 
#                     "timestamp": (base_time - timedelta(hours=1)).isoformat()
#                 },
#                 {
#                     "user_id": 2, 
#                     "job_id": 6, 
#                     "action_type": "SAVE", 
#                     "timestamp": (base_time - timedelta(hours=3)).isoformat()
#                 },
                
#                 # User 3 interactions
#                 {
#                     "user_id": 3, 
#                     "job_id": 5, 
#                     "action_type": "VIEW", 
#                     "timestamp": (base_time - timedelta(hours=4)).isoformat()
#                 },
#                 {
#                     "user_id": 3, 
#                     "job_id": 8, 
#                     "action_type": "APPLY", 
#                     "timestamp": (base_time - timedelta(days=2)).isoformat()
#                 }
#             ]
            
#             jobs_df = pd.DataFrame(jobs)
#             ratings_df = pd.DataFrame(ratings)
#             activity_logs_df = pd.DataFrame(activity_logs)
            
#             cls._jobs = jobs_df
#             cls._instance = create_hybrid_recommender(
#                 jobs_df, 
#                 ratings_df, 
#                 activity_logs_df
#             )
#         return cls._instance, cls._jobs
    
#     @classmethod
#     def get_job_details(cls, job_id, jobs_df):
#         job = jobs_df[jobs_df['job_id'] == job_id].to_dict('records')
#         return job[0] if job else None

# # Dependency injection for recommender
# def get_recommender():
#     return RecommenderManager.get_recommender()

# @app.post("/recommend", response_model=List[RecommendationResponse])
# async def get_recommendations(
#     request: RecommendationRequest, 
#     recommender_and_jobs = Depends(get_recommender)
# ):
#     try:
#         # Unpack the recommender and jobs dataframe
#         recommender, jobs_df = recommender_and_jobs
        
#         # Lấy recommendation
#         recommendations = recommender.recommend(
#             user_id=request.user_id,
#             top_n=request.top_n
#         )
        
#         # Merge recommendation scores with job details
#         detailed_recommendations = []
#         for rec in recommendations:
#             job_details = RecommenderManager.get_job_details(rec['job_id'], jobs_df)
#             if job_details:
#                 detailed_rec = {
#                     **rec,
#                     "title": job_details['title'],
#                     "description": job_details['description'],
#                     "company": job_details.get('company'),
#                     "location": job_details.get('location'),
#                     "salary_range": job_details.get('salary_range')
#                 }
#                 detailed_recommendations.append(detailed_rec)
        
#         return detailed_recommendations
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Các endpoint khác giữ nguyên
# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "version": "1.0.0"}

# @app.post("/refresh")
# async def refresh_recommender():
#     RecommenderManager._instance = None
#     return {"status": "recommender refreshed"}

# # Chạy ứng dụng
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)