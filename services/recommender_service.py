import pandas as pd
from functools import lru_cache
from models.models import Job, Rating, ActivityLog
from sqlalchemy.orm import Session
from models.hybrid_model import create_hybrid_recommender

class RecommenderManager:
    @staticmethod
    @lru_cache(maxsize=1)
    def get_recommender(
        db: Session, 
        limit=1000, 
        offset=0, 
        filters=None,
        content_weight=0.3,
        collaborative_weight=0.4,
        activity_weight=0.3,
        min_ratings=2,
        rating_scale=1.0
    ):
        """
        Tạo và lưu trữ bộ gợi ý với các tham số tùy chỉnh
        
        :param db: Session cơ sở dữ liệu
        :param limit: Giới hạn số lượng công việc lấy từ DB
        :param offset: Vị trí bắt đầu lấy dữ liệu
        :param filters: Dict chứa các bộ lọc (category_id, level_id, min_salary, location, job_ids)
        :param content_weight: Trọng số cho gợi ý dựa trên nội dung
        :param collaborative_weight: Trọng số cho gợi ý cộng tác
        :param activity_weight: Trọng số cho hoạt động người dùng
        :param min_ratings: Số lượng đánh giá tối thiểu để xem xét
        :param rating_scale: Giá trị mặc định cho rating
        :return: (HybridRecommender, DataFrame công việc)
        """
        try:
            # Khởi tạo truy vấn cơ bản
            jobs_query = db.query(Job).filter(Job.status == "ACTIVE")
            
            # Áp dụng các bộ lọc (nếu có)
            if filters:
                if 'category_id' in filters:
                    jobs_query = jobs_query.filter(Job.category_id.in_(filters['category_id']))
                if 'level_id' in filters:
                    jobs_query = jobs_query.filter(Job.level_id.in_(filters['level_id']))
                if 'min_salary' in filters:
                    jobs_query = jobs_query.filter(Job.job_max_salary >= filters['min_salary'])
                if 'location' in filters:
                    jobs_query = jobs_query.filter(Job.job_location.ilike(f"%{filters['location']}%"))
            
            # Áp dụng phân trang
            jobs = jobs_query.limit(limit).offset(offset).all()
            
            # Tối ưu hóa truy vấn nhà tuyển dụng bằng cách chỉ lấy các employer_id cần thiết
            employer_ids = [job.employer_id for job in jobs]
            employers_query = db.query(Job.employer.property.mapper.class_).filter(
                Job.employer.property.mapper.class_.employer_id.in_(employer_ids)
            )
            
            employers = {
                employer.employer_id: employer.company_logo 
                for employer in employers_query.all()
            }
            company_names = {
                employer.employer_id: employer.company_name 
                for employer in employers_query.all()
            }
            
            jobs_data = [
                {
                    "job_id": job.job_id,
                    "job_title": job.job_title,
                    "logo": employers.get(job.employer_id, None),
                    "company_name": company_names.get(job.employer_id, None),
                    "job_description": job.job_description,
                    "job_location": job.job_location,
                    "job_min_salary": job.job_min_salary,
                    "job_max_salary": job.job_max_salary,
                    "category_id": job.category_id,
                    "level_id": job.level_id
                } for job in jobs
            ]
            
            # Lấy và chuẩn bị dữ liệu ratings
            ratings_query = db.query(Rating)
            # Nếu có bộ lọc về job_id, chỉ lấy ratings liên quan
            if filters and 'job_ids' in filters:
                ratings_query = ratings_query.filter(Rating.job_id.in_(filters['job_ids']))
            
            ratings = ratings_query.all()
            ratings_data = [
                {
                    "user_id": rating.student_id,
                    "job_id": rating.job_id,
                    "rating": rating_scale  # Sử dụng tham số thay vì giá trị cứng
                } for rating in ratings
            ]
            
            # Tương tự với activity logs
            activity_logs_query = db.query(ActivityLog)
            if filters and 'job_ids' in filters:
                activity_logs_query = activity_logs_query.filter(ActivityLog.job_id.in_(filters['job_ids']))
            
            activity_logs = activity_logs_query.all()
            activity_logs_data = [
                {
                    "user_id": log.user_id,
                    "job_id": log.job_id,
                    "action_type": log.action_type
                } for log in activity_logs
            ]
            
            # Chuyển đổi thành DataFrame
            jobs_df = pd.DataFrame(jobs_data)
            ratings_df = pd.DataFrame(ratings_data)
            activity_logs_df = pd.DataFrame(activity_logs_data)
            
            print("Jobs DataFrame:", jobs_df.head(2))
            print(f"Ratings DataFrame: {len(ratings_df)} rows")
            print(f"Activity Logs DataFrame: {len(activity_logs_df)} rows")
            
            # Tạo recommender với các tham số tùy chỉnh
            recommender = create_hybrid_recommender(
                jobs_df, 
                ratings_df, 
                activity_logs_df,
                content_weight=content_weight,
                collaborative_weight=collaborative_weight,
                activity_weight=activity_weight,
                min_ratings=min_ratings
            )
            
            return recommender, jobs_df
            
        except Exception as e:
            print(f"Error in get_recommender: {e}")
            # Trả về DataFrame trống trong trường hợp lỗi
            return None, pd.DataFrame()

    @staticmethod
    def get_job_details(job_id, jobs_df):
        """
        Lấy thông tin chi tiết về một công việc từ DataFrame
        
        :param job_id: ID công việc
        :param jobs_df: DataFrame chứa thông tin công việc
        :return: Thông tin chi tiết về công việc
        """
        try:
            job = jobs_df[jobs_df['job_id'] == job_id].to_dict('records')
            return job[0] if job else None
        except Exception as e:
            print(f"Error in get_job_details: {e}")
            return None

    @classmethod
    def invalidate_cache(cls, specific_args=None):
        """
        Xóa bộ nhớ đệm của phương thức get_recommender.
        
        :param specific_args: Tuple chứa các tham số cụ thể để xóa cache.
                             Nếu None, xóa toàn bộ cache.
        """
        try:
            if specific_args:
                # Trong thực tế, đây không phải là cách lru_cache hoạt động
                # Chúng ta không thể xóa một entry cụ thể từ cache
                # Vì vậy, chúng ta xóa toàn bộ cache
                cls.get_recommender.cache_clear()
            else:
                cls.get_recommender.cache_clear()
        except Exception as e:
            print(f"Error in invalidate_cache: {e}")

    @classmethod
    def get_cache_info(cls):
        """
        Trả về thông tin về trạng thái hiện tại của cache.
        
        :return: CacheInfo object
        """
        try:
            return cls.get_recommender.cache_info()
        except Exception as e:
            print(f"Error in get_cache_info: {e}")
            return None