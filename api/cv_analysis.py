from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import os
import logging

from models.database import get_db
from models.schemas import (
    CVAnalysisRequest, 
    CVAnalysisResponse, 
    CVRecommendationRequest, 
    CVRecommendationResponse,
    CVJobMatchResponse,
    ResumeRecommendationResponse
)
from services.cv_analyzer_service import EnhancedCVAnalyzer, EnhancedCVRecommender

# Tạo router với tiền tố /api/cv
router = APIRouter(prefix="/cv", tags=["CV Analysis"])

# Khởi tạo các dịch vụ với mô hình nâng cao
cv_analyzer = EnhancedCVAnalyzer(temp_upload_dir="uploads")
cv_recommender = EnhancedCVRecommender(cv_analyzer=cv_analyzer)

logger = logging.getLogger(__name__)

@router.post("/analyze/{resume_id}", response_model=CVAnalysisResponse)
async def analyze_cv(
    resume_id: int,
    db: Session = Depends(get_db)
):
    """
    Phân tích CV của sinh viên theo resume_id
    """
    try:
        from models.models import ResumeCV
        
        # Truy vấn thông tin CV từ database
        resume = db.query(ResumeCV).filter(ResumeCV.resume_id == resume_id).first()
        if not resume:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": f"Resume with ID {resume_id} not found"}
            )
            
        # Phân tích CV với mô hình nâng cao
        analysis_result = cv_analyzer.analyze_cv(resume.resume_file)
        
        # Chuyển đổi kết quả phân tích thành response format
        response_data = {
            "success": analysis_result['success'],
            "resume_id": resume_id,
            "skills": analysis_result.get('skills', []),
            "education": analysis_result.get('education', []),
            "experience": analysis_result.get('experience', []),
            "summary": analysis_result.get('summary', ""),
            "industry": analysis_result.get('industry', ""),
            "seniority": analysis_result.get('seniority', ""),
            "language": analysis_result.get('language', "en")
        }
        
        if not analysis_result['success']:
            response_data["error"] = analysis_result.get('error', 'Unknown error occurred')
        
        return response_data
    
    except Exception as e:
        logger.error(f"Error analyzing CV: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error analyzing CV: {str(e)}"}
        )

@router.post("/upload", response_model=CVAnalysisResponse)
async def upload_and_analyze_cv(
    file: UploadFile = File(...),
    student_id: Optional[int] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload và phân tích CV mới.
    Nếu cung cấp student_id, CV sẽ được lưu vào database.
    """
    try:
        # Tạo thư mục uploads nếu chưa tồn tại
        os.makedirs(cv_analyzer.temp_upload_dir, exist_ok=True)
        
        # Lưu file tạm thời
        file_path = os.path.join(cv_analyzer.temp_upload_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Phân tích CV với mô hình nâng cao
        analysis_result = cv_analyzer.analyze_cv(file_path)
        
        # Nếu có student_id, lưu CV vào database
        if student_id and analysis_result['success']:
            from models.models import ResumeCV
            from datetime import datetime
            
            # Tạo tên file CV mới (đảm bảo không trùng lặp)
            from uuid import uuid4
            
            # Lấy phần mở rộng của file
            file_ext = os.path.splitext(file.filename)[1]
            new_filename = f"cv_{student_id}_{uuid4().hex}{file_ext}"
            new_filepath = os.path.join("uploads", new_filename)
            
            # Di chuyển file tạm thời đến vị trí mới
            import shutil
            shutil.move(file_path, new_filepath)
            
            # Tạo bản ghi ResumeCV mới
            new_resume = ResumeCV(
                student_id=student_id,
                resume_title=f"CV_{datetime.now().strftime('%Y%m%d')}",
                resume_file=new_filepath,
                resume_description=analysis_result.get('summary', "Uploaded via API") or "Uploaded via API",
                is_active=1,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            db.add(new_resume)
            db.commit()
            db.refresh(new_resume)
            
            # Cập nhật kết quả phân tích với resume_id
            analysis_result['resume_id'] = new_resume.resume_id
        
        # Chuyển đổi kết quả phân tích thành response format
        response_data = {
            "success": analysis_result['success'],
            "resume_id": analysis_result.get('resume_id', None),
            "skills": analysis_result.get('skills', []),
            "education": analysis_result.get('education', []),
            "experience": analysis_result.get('experience', []),
            "summary": analysis_result.get('summary', ""),
            "industry": analysis_result.get('industry', ""),
            "seniority": analysis_result.get('seniority', ""),
            "language": analysis_result.get('language', "en")
        }
        
        if not analysis_result['success']:
            response_data["error"] = analysis_result.get('error', 'Unknown error occurred')
            
        return response_data
    
    except Exception as e:
        logger.error(f"Error uploading and analyzing CV: {str(e)}")
        # Xóa file tạm nếu có lỗi
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
            
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error uploading and analyzing CV: {str(e)}"}
        )

@router.get("/recommend/resume/{resume_id}", response_model=CVRecommendationResponse)
async def recommend_jobs_by_resume(
    resume_id: int,
    limit: Optional[int] = 10,
    db: Session = Depends(get_db)
):
    """
    Đề xuất công việc phù hợp với CV sử dụng mô hình ML nâng cao
    """
    try:
        # Đề xuất công việc với mô hình nâng cao
        recommendations = cv_recommender.recommend_jobs_for_cv(
            cv_id=resume_id,
            db=db,
            limit=limit
        )
        
        # Bổ sung thông tin chi tiết cho mỗi công việc
        from models.models import Job, Employer
        
        enhanced_recommendations = []
        for rec in recommendations:
            job_id = rec['job_id']
            
            # Truy vấn thông tin chi tiết về công việc và công ty
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                employer = db.query(Employer).filter(Employer.employer_id == job.employer_id).first()
                
                # Ensure all required fields are present
                job_match = {
                    'job_id': rec['job_id'],
                    'job_title': rec['job_title'],
                    'match_score': rec['match_score'],
                    'skill_match_score': rec.get('skill_match_score', 0.0),
                    'content_similarity': rec.get('content_similarity', 0.0),
                    'matched_skills': rec.get('matched_skills', []),
                    'missing_skills': rec.get('missing_skills', []),
                    'employer_id': rec.get('employer_id', job.employer_id),
                    'category_id': rec.get('category_id', job.category_id),
                    'level_id': rec.get('level_id', job.level_id),
                    'reason': rec.get('reason', ''),
                    'logo': employer.company_logo if employer else '',
                    'company_name': employer.company_name if employer else '',
                    'job_location': job.job_location or '',
                    'job_min_salary': float(job.job_min_salary or 0),
                    'job_max_salary': float(job.job_max_salary or 0)
                }
                
                enhanced_recommendations.append(job_match)
                #logger.info(f"Enhanced Recommendation: {job_match}")
        
        # Create a valid response that matches the Pydantic model
        response = {"recommendations": enhanced_recommendations}
        return response
    
    except Exception as e:
        logger.error(f"Error recommending jobs by resume: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error recommending jobs: {str(e)}"
        )

@router.get("/recommend/student/{student_id}", response_model=CVRecommendationResponse)
async def recommend_jobs_by_student(
    student_id: int,
    limit: Optional[int] = 10,
    db: Session = Depends(get_db)
):
    """
    Đề xuất công việc phù hợp cho sinh viên dựa trên CV gần đây nhất sử dụng mô hình ML nâng cao
    """
    try:
        # Tìm kiếm các công việc phù hợp với CV của sinh viên sử dụng mô hình nâng cao
        recommendations = cv_recommender.recommend_jobs_for_student(
            student_id=student_id,
            db=db,
            limit=limit
        )
        
        # Bổ sung thông tin chi tiết cho mỗi công việc
        from models.models import Job, Employer
        
        enhanced_recommendations = []
        for rec in recommendations:
            job_id = rec['job_id']
            
            # Truy vấn thông tin chi tiết về công việc và công ty
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                employer = db.query(Employer).filter(Employer.employer_id == job.employer_id).first()
                
                # Ensure all required fields are present
                job_match = {
                    'job_id': rec['job_id'],
                    'job_title': rec['job_title'],
                    'match_score': rec['match_score'],
                    'skill_match_score': rec.get('skill_match_score', 0.0),
                    'content_similarity': rec.get('content_similarity', 0.0),
                    'matched_skills': rec.get('matched_skills', []),
                    'missing_skills': rec.get('missing_skills', []),
                    'employer_id': rec.get('employer_id', job.employer_id),
                    'category_id': rec.get('category_id', job.category_id),
                    'level_id': rec.get('level_id', job.level_id),
                    'reason': rec.get('reason', ''),
                    'logo': employer.company_logo if employer else '',
                    'company_name': employer.company_name if employer else '',
                    'job_location': job.job_location or '',
                    'job_min_salary': float(job.job_min_salary or 0),
                    'job_max_salary': float(job.job_max_salary or 0)
                }
                
                enhanced_recommendations.append(job_match)
                logger.info(f"Enhanced Recommendation: {job_match}")
        
        # Create a valid response that matches the Pydantic model
        response = {"recommendations": enhanced_recommendations}
        logger.info(f"Returning {len(enhanced_recommendations)} recommendations")
        return response
    
    except Exception as e:
        logger.error(f"Error recommending jobs by student: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error recommending jobs: {str(e)}"
        )

@router.get("/skills", response_model=List[str])
async def get_available_skills():
    """
    Lấy danh sách kỹ năng được hỗ trợ bởi hệ thống phân tích CV
    """
    try:
        return cv_analyzer.TECHNICAL_SKILLS
    except Exception as e:
        logger.error(f"Error getting skills list: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving skills list: {str(e)}"
        )

@router.get("/job/{job_id}/matching_resumes", response_model=ResumeRecommendationResponse)
async def recommend_resumes_for_job(
    job_id: int,
    limit: int,
    db: Session = Depends(get_db)
):
    """
    Lọc n CV phù hợp nhất với một job từ tổng số CV đã apply vào job
    """
    try:
        from models.models import Job, Application, ResumeCV, Student
        from sqlalchemy import desc

        # Kiểm tra job có tồn tại không
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job with ID {job_id} not found"
            )

        # Lấy danh sách các CV đã apply vào job
        applications = db.query(Application).filter(
            Application.job_id == job_id
        ).all()

        if not applications:
            return {"recommendations": []}        # Lấy danh sách resume_id từ applications và map với application_id tương ứng
        resume_app_map = {app.resume_id: app.application_id for app in applications if app.resume_id}
        resume_ids = list(resume_app_map.keys())
        
        if not resume_ids:
            return {"recommendations": []}

        # Phân tích matching score cho từng CV
        resume_matches = []
        
        for resume_id in resume_ids:
            # Truy vấn thông tin CV từ database
            resume = db.query(ResumeCV).filter(ResumeCV.resume_id == resume_id ).first()
            if not resume:
                continue
                
            # Lấy application_id tương ứng với resume_id
            application_id = resume_app_map.get(resume_id)

            # Phân tích CV để lấy thông tin
            cv_analysis = cv_analyzer.analyze_cv(resume.resume_file)
            if not cv_analysis.get('success', False):
                continue

            # Chuẩn bị dữ liệu job để matching
            job_desc = f"{job.job_title} {job.job_description or ''} {job.job_requirements or ''}"
            
            # Lấy danh sách kỹ năng của job
            job_skills = cv_recommender.get_job_skills(job_id, db)
            if not job_skills and job.job_requirements:
                job_skills = cv_recommender.extract_skills_from_job(job.job_requirements)
            
            # Tính toán match score giữa CV và job
            match_result = cv_recommender.match_cv_to_job(cv_analysis, job_desc, job_skills)
            
            # Lấy thông tin student
            student = db.query(Student).filter(Student.student_id == resume.student_id).first()
            student_name = f"{student.user.first_name} {student.user.last_name}" if student else "Unknown"
            student_image = student.profile_image if student and student.user else None
              # Tạo resume match response
            resume_match = {
                'resume_id': resume_id,
                'application_id': application_id,  # Thêm application_id vào response
                'student_name': student_name,
                'profile_image': student_image,
                'match_score': match_result['match_score'],
                'skill_match_score': match_result['skill_match_score'],
                'content_similarity': match_result['content_similarity'],
                'matched_skills': match_result['matched_skills'],
                'missing_skills': match_result['missing_skills'],
                'resume_title': resume.resume_title,
                'reason': cv_recommender._generate_match_reason(match_result)
            }
            
            resume_matches.append(resume_match)

        # Sắp xếp theo match_score và giới hạn số lượng
        sorted_matches = sorted(resume_matches, key=lambda x: x['match_score'], reverse=True)[:limit]
        
        # Tạo response
        response = {"recommendations": sorted_matches}
        logger.info(f"Found {len(sorted_matches)} matching resumes for job {job_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recommending resumes for job: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error recommending resumes for job: {str(e)}"
        )