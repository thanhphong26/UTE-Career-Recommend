from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base

class Job(Base):
    __tablename__ = "jobs"
    
    job_id = Column(Integer, primary_key=True, index=True)
    employer_id = Column(Integer, ForeignKey("employers.employer_id"), nullable=False)
    category_id = Column(Integer, ForeignKey("job_categories.category_id"), nullable=True)
    level_id = Column(Integer, ForeignKey("job_levels.level_id"), nullable=True)
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
    employer = relationship("Employer", back_populates="jobs")
    category = relationship("JobCategory", back_populates="jobs")
    level = relationship("JobLevel", back_populates="jobs")

class Rating(Base):
    __tablename__ = "saved_jobs"
    student_id = Column(Integer, ForeignKey("students.student_id"), primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.job_id"), primary_key=True, index=True)

class ActivityLog(Base):
    __tablename__ = "user_activity_logs"
    
    activity_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    job_id = Column(Integer, ForeignKey("jobs.job_id"), nullable=False)
    action_type = Column(String(50), nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), nullable=False)
    password = Column(String(500), nullable=False)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    gender = Column(Integer, nullable=True)
    dob = Column(DateTime, nullable=True)
    phone_number = Column(String(20), nullable=True)
    address = Column(String(1000), nullable=True)
    province_id = Column(Integer, nullable=True)
    district_id = Column(Integer, nullable=True)
    ward_id = Column(Integer, nullable=True)
    is_active = Column(Integer, nullable=False)
    reason_blocked = Column(String(1000), nullable=True)
    role_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    google_account_id = Column(String(255), nullable=True)

class Employer(Base):
    __tablename__ = "employers"
    
    employer_id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String(500), nullable=False)
    company_address = Column(String(1000), nullable=True)
    company_logo = Column(String(1000), nullable=True)
    company_email = Column(String(100), nullable=False)
    company_website = Column(String(1000), nullable=True)
    company_description = Column(String(10000), nullable=True)  
    background_image = Column(String(1000), nullable=True)
    video_introduction = Column(String(1000), nullable=True)
    company_size = Column(String(100), nullable=True)
    approval_status = Column(String(50), nullable=False)  
    rejected_reason = Column(String(500), nullable=True)
    business_certificate = Column(String(1000), nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    industry_id = Column(Integer, nullable=True)
    jobs = relationship("Job", back_populates="employer")

class Student(Base):
    __tablename__ = "students"
    
    student_id = Column(Integer, primary_key=True, index=True)
    university_email = Column(String(255), nullable=False)
    year = Column(Integer, nullable=True)
    profile_image = Column(String(1000), nullable=True)
    is_find = Column(Integer, nullable=False)  
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    category_id = Column(Integer, ForeignKey("job_categories.category_id"), nullable=True)

class Skills(Base):
    __tablename__ = "skills"
    
    skill_id = Column(Integer, primary_key=True, index=True)
    skill_name = Column(String(255), nullable=False)
    skill_description = Column(String(1000), nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class JobSkill(Base):
    __tablename__ = "job_skills"
    
    job_skill_id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.job_id"), nullable=False)
    skill_id = Column(Integer, ForeignKey("skills.skill_id"), nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class JobCategory(Base):
    __tablename__ = "job_categories"
    
    category_id = Column(Integer, primary_key=True, index=True)
    category_name = Column(String(255), nullable=False)
    is_active = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    jobs = relationship("Job", back_populates="category")

class JobLevel(Base):
    __tablename__ = "job_levels"
    
    level_id = Column(Integer, primary_key=True, index=True)
    name_level = Column(String(255), nullable=False)
    is_active = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    jobs = relationship("Job", back_populates="level")

class ResumeCV(Base):
    __tablename__ = "resumes"
    
    resume_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.student_id"), nullable=False)
    resume_title = Column(String(255), nullable=False)
    resume_file = Column(String(1000), nullable=False)
    resume_description = Column(String(10000), nullable=True)
    level_id = Column(Integer, ForeignKey("job_levels.level_id"), nullable=True)
    is_active = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
class Application(Base):
    __tablename__ = "applications"
    
    application_id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.job_id"), nullable=False)
    resume_id = Column(Integer, ForeignKey("resumes.resume_id"), nullable=True)
    application_status = Column(String(50), nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)