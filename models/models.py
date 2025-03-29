from sqlalchemy import Column, Integer, String, Float, DateTime
from .database import Base

class Job(Base):
    __tablename__ = "jobs"
    
    job_id = Column(Integer, primary_key=True, index=True)
    employer_id = Column(Integer)
    category_id = Column(Integer)
    level_id = Column(Integer)
    job_title = Column(String(255), nullable=False)
    job_description = Column(String(1000), nullable=True)
    job_requirements = Column(String(1000), nullable=True)
    job_location = Column(String(255), nullable=True)
    job_min_salary = Column(Float(precision=18), nullable=True)
    job_max_salary = Column(Float(precision=18), nullable=True)
    job_deadline = Column(DateTime, nullable=True)
    amount = Column(Integer, nullable=True)
    status = Column(String(50), nullable=False)
    rejection_reason = Column(String(255), nullable=True)
    package_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
class Rating(Base):
    __tablename__ = "saved_jobs"
    student_id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, primary_key=True, index=True)

class ActivityLog(Base):
    __tablename__ = "activity_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    job_id = Column(Integer, nullable=False)
    action_type = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False)