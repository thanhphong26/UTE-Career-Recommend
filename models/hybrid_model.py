import pandas as pd
import numpy as np
from datetime import datetime, timezone
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional, Dict


class HybridRecommender:
    def __init__(
        self, 
        jobs_df, 
        ratings_df, 
        activity_logs_df=None, 
        job_levels_df=None, 
        job_categories_df=None, 
        job_skills_df=None,
        content_weight=0.3,
        collaborative_weight=0.4,
        activity_weight=0.3
    ):
        """
        Khởi tạo recommender với nhiều kỹ thuật
        
        """
        self.jobs_df = jobs_df
        self.ratings_df = ratings_df
        self.activity_logs_df = activity_logs_df
        self.job_levels_df = job_levels_df
        self.job_categories_df = job_categories_df
        self.job_skills_df = job_skills_df
        self.user_id = None  # Will be set when recommend is called
        
        # Xây dựng ma trận đánh giá để collaborative filtering
        self._build_rating_matrix()
        
        # Xây dựng vector đặc trưng cho content-based filtering nếu có đủ dữ liệu
        if self.job_levels_df is not None and self.job_categories_df is not None and self.job_skills_df is not None:
            self._build_content_vectors()
        else:
            print("Warning: Missing job metadata dataframes, content-based filtering will be based on job descriptions only")
            self._build_simplified_content_vectors()
        
        # Trọng số cho từng phương pháp recommendation
        self.weights = {
            'collaborative': collaborative_weight,
            'content_based': content_weight,
            'activity_signal': activity_weight
        }
    
    def _build_rating_matrix(self):
        """
        Xây dựng ma trận đánh giá để collaborative filtering
        """
        if self.ratings_df.empty:
            print("Warning: Ratings DataFrame is empty, collaborative filtering will be limited")
            self.rating_matrix = None
            self.user_ids = []
            self.job_ids = []
            return
            
        try:
            user_job_matrix = self.ratings_df.pivot_table(
                index='user_id', 
                columns='job_id', 
                values='rating',
                fill_value=0
            )
            
            self.rating_matrix = csr_matrix(user_job_matrix.values)
            self.user_ids = user_job_matrix.index.tolist()
            self.job_ids = user_job_matrix.columns.tolist()
        except Exception as e:
            print(f"Error building rating matrix: {e}")
            self.rating_matrix = None
            self.user_ids = []
            self.job_ids = []
    
    def _build_simplified_content_vectors(self):
        """
        Xây dựng vector đặc trưng cho content-based filtering chỉ dựa trên tiêu đề công việc và vị trí
        """
        try:
            # Sử dụng tiêu đề công việc để tạo nội dung
            content_data = self.jobs_df['job_title'].fillna('')
            
            # Nếu có job_location, thêm vào nội dung
            if 'job_location' in self.jobs_df.columns:
                content_data = content_data + ' ' + self.jobs_df['job_location'].fillna('')
                
            # Sử dụng TF-IDF để vector hóa nội dung
            tfidf = TfidfVectorizer(stop_words='english')
            self.content_matrix = tfidf.fit_transform(content_data)
        except Exception as e:
            print(f"Error building simplified content vectors: {e}")
            self.content_matrix = None
    
    def _build_content_vectors(self):
        """
        Xây dựng vector đặc trưng cho content-based filtering
        """
        # Kiểm tra xem có đủ dataframes cần thiết không
        if self.job_levels_df is None or self.job_categories_df is None or self.job_skills_df is None:
            print("Cannot build content vectors: missing required dataframes")
            return
            
        try:
            # Kết hợp các thuộc tính liên quan từ các bảng liên kết để tạo vector
            jobs_with_details = self.jobs_df.copy()
            
            # Merge với job_levels_df
            if 'level_id' in jobs_with_details.columns:
                jobs_with_details = jobs_with_details.merge(
                    self.job_levels_df, 
                    left_on='level_id', 
                    right_on='level_id', 
                    how='left'
                )
            elif 'job_level_id' in jobs_with_details.columns:
                jobs_with_details = jobs_with_details.merge(
                    self.job_levels_df, 
                    left_on='job_level_id', 
                    right_on='level_id', 
                    how='left'
                )
            
            # Merge với job_categories_df
            if 'category_id' in jobs_with_details.columns:
                jobs_with_details = jobs_with_details.merge(
                    self.job_categories_df, 
                    left_on='category_id', 
                    right_on='category_id', 
                    how='left'
                )
            elif 'job_category_id' in jobs_with_details.columns:
                jobs_with_details = jobs_with_details.merge(
                    self.job_categories_df, 
                    left_on='job_category_id', 
                    right_on='category_id', 
                    how='left'
                )
            
            # Thêm tên các cột sau khi merge để tạo nội dung
            level_name_col = 'name_level' if 'name_level' in jobs_with_details.columns else 'job_level_name'
            category_name_col = 'category_name' if 'category_name' in jobs_with_details.columns else 'job_category_name'
            
            print(f"Level name column: {level_name_col}")
            print(f"Category name column: {category_name_col}")
            # Tạo cột 'content' bằng cách kết hợp các thuộc tính liên quan
            jobs_with_details['content'] = jobs_with_details['job_title']
            
            # Thêm thông tin level nếu có
            if level_name_col in jobs_with_details.columns:
                jobs_with_details['content'] += ' ' + jobs_with_details[level_name_col].fillna('')
            
            # Thêm thông tin category nếu có
            if category_name_col in jobs_with_details.columns:
                jobs_with_details['content'] += ' ' + jobs_with_details[category_name_col].fillna('')
            
            # Sử dụng TF-IDF để vector hóa nội dung
            tfidf = TfidfVectorizer(stop_words='english')
            self.content_matrix = tfidf.fit_transform(jobs_with_details['content'])
        except Exception as e:
            print(f"Error building content vectors: {e}")
            # Fallback để không làm crash chương trình
            self._build_simplified_content_vectors()
    
    def _collaborative_recommendations(self, user_id, top_n):
        """
        Recommendation dựa trên hành vi người dùng tương tự
        
        :param user_id: ID người dùng
        :param top_n: Số lượng recommendation
        :return: Danh sách recommendation
        """
        if self.rating_matrix is None or len(self.user_ids) == 0:
            return []
            
        try:
            if user_id not in self.user_ids:
                return []
                
            user_index = self.user_ids.index(user_id)
            user_ratings = self.rating_matrix[user_index]
            
            # Tính điểm tương đồng với các công việc đã được đánh giá
            sim_scores = cosine_similarity(user_ratings, self.rating_matrix).flatten()
            similar_users_indices = sim_scores.argsort()[::-1][1:6]  # Top 5 người dùng tương tự
            print(f"Similar users indices: {similar_users_indices}")

            recommendations = {}
            for idx in similar_users_indices:
                similar_user_id = self.user_ids[idx]
                user_job_ratings = self.ratings_df[
                    self.ratings_df['user_id'] == similar_user_id
                ]
                
                # Lấy công việc đã được đánh giá bởi người dùng hiện tại
                user_rated_jobs = set(
                    self.ratings_df[self.ratings_df['user_id'] == user_id]['job_id']
                )
                
                for _, row in user_job_ratings.iterrows():
                    job_id = row['job_id']
                    if job_id not in user_rated_jobs:
                        recommendations[job_id] = recommendations.get(
                            job_id, 0
                        ) + row.get('rating', 1.0) * sim_scores[idx]
            
            return [
                {'job_id': job_id, 'score': score} 
                for job_id, score in sorted(
                    recommendations.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:top_n]
            ]
        except Exception as e:
            print(f"Collaborative filtering error: {e}")
            return []
    
    def _content_based_recommendations(self, user_id, top_n):
        """
        Recommendation dựa trên nội dung công việc
        
        :param user_id: ID người dùng
        :param top_n: Số lượng recommendation
        :return: Danh sách recommendation
        """
        try:
            # Kiểm tra nếu content_matrix không được khởi tạo
            if not hasattr(self, 'content_matrix') or self.content_matrix is None:
                return []
                
            # Lấy các công việc đã được đánh giá bởi người dùng
            user_rated_jobs = self.ratings_df[
                self.ratings_df['user_id'] == user_id
            ]['job_id']
            
            if len(user_rated_jobs) == 0:
                return []
            
            # Tìm các index của công việc đã được đánh giá
            job_id_to_index = {job_id: i for i, job_id in enumerate(self.jobs_df['job_id'])}
            rated_indices = [
                job_id_to_index.get(job_id) 
                for job_id in user_rated_jobs 
                if job_id in job_id_to_index
            ]
            
            # Loại bỏ các None values
            rated_indices = [idx for idx in rated_indices if idx is not None]
            
            if not rated_indices:
                return []
                
            # Tạo user profile từ các công việc đã được đánh giá
            user_profile = np.asarray(
                self.content_matrix[rated_indices].mean(axis=0)
            ).reshape(1, -1)

            content_sim_scores = cosine_similarity(
                user_profile, 
                self.content_matrix
            ).flatten()

            # Lấy các ID công việc đã được đánh giá
            rated_job_ids = set(user_rated_jobs)
            
            # Tạo danh sách recommendation 
            recommendations = []
            for i, (job_id, score) in enumerate(zip(self.jobs_df['job_id'], content_sim_scores)):
                if job_id not in rated_job_ids:
                    recommendations.append({
                        'job_id': job_id, 
                        'score': score
                    })
            
            return sorted(
                recommendations, 
                key=lambda x: x['score'], 
                reverse=True
            )[:top_n]
        except Exception as e:
            print(f"Content-based filtering error: {e}")
            return []
    
    def _activity_signal_recommendations(self, user_id, top_n):
        """
        Recommendation dựa trên tín hiệu hoạt động
        
        :param user_id: ID người dùng
        :param top_n: Số lượng recommendation
        :return: Danh sách recommendation
        """
        if self.activity_logs_df is None or self.activity_logs_df.empty or 'user_id' not in self.activity_logs_df.columns:
            return []
        
        try:
            # Phân tích nhật ký hoạt động của người dùng
            user_logs = self.activity_logs_df[
                self.activity_logs_df['user_id'] == user_id
            ]
            
            if user_logs.empty:
                return []
                
            # Tính điểm dựa trên loại hoạt động
            def calculate_activity_score(action_type):
                # Định nghĩa trọng số cho các loại hoạt động
                if action_type == 'VIEW':
                    return 1.0
                elif action_type == 'CLICK':
                    return 2.0
                elif action_type == 'APPLY':
                    return 3.0
                else:
                    return 0.5
                    
            # Tính tổng điểm cho mỗi công việc
            activity_scores = {}
            for _, row in user_logs.iterrows():
                job_id = row['job_id']
                action_type = row['action_type']
                score = calculate_activity_score(action_type)
                activity_scores[job_id] = activity_scores.get(job_id, 0) + score
                
            # Lấy danh sách công việc đã được đánh giá
            rated_jobs = set(
                self.ratings_df[self.ratings_df['user_id'] == user_id]['job_id']
            )
            
            # Lọc ra các công việc chưa được đánh giá
            recommendations = [
                {
                    'job_id': job_id, 
                    'score': score
                } 
                for job_id, score in activity_scores.items()
                if job_id not in rated_jobs
            ]
           
            return sorted(
                recommendations, 
                key=lambda x: x['score'], 
                reverse=True
            )[:top_n]
        except Exception as e:
            print(f"Activity signal filtering error: {e}")
            return []
    
    def _popular_recommendations(self, top_n):
        # Đề xuất công việc phổ biến nhất dựa trên số lượng lượt xem/ứng tuyển
        try:
            if self.activity_logs_df is None or self.activity_logs_df.empty:
                # Nếu không có dữ liệu hoạt động, đề xuất ngẫu nhiên các công việc mới nhất
                return [
                    {'job_id': job_id, 'score': 1.0} 
                    for job_id in self.jobs_df['job_id'].sample(min(top_n, len(self.jobs_df)))
                ]
                
            # Đếm số hoạt động cho mỗi công việc
            job_counts = self.activity_logs_df['job_id'].value_counts().reset_index()
            job_counts.columns = ['job_id', 'count']
            
            # Lấy top N công việc phổ biến nhất
            popular_jobs = job_counts.sort_values('count', ascending=False).head(top_n * 2)
            
            recommendations = []
            for _, row in popular_jobs.iterrows():
                job_id = row['job_id']
                job_data = self.jobs_df[self.jobs_df['job_id'] == job_id]
                if not job_data.empty:
                    job_details = {'job_id': job_id, 'score': row['count']}
                    
                    # Thêm các thông tin chi tiết
                    for col in job_data.columns:
                        if col != 'job_id':
                            job_details[col] = job_data[col].values[0]
                            
                    recommendations.append(job_details)
            
            return recommendations[:top_n]
        except Exception as e:
            print(f"Popular recommendations error: {e}")
            return []
            
    def _get_recommendation_reason(self, primary_source, score_components):
        """
        Tạo lý do đề xuất dựa trên nguồn điểm chính
        
        :param primary_source: Nguồn chính của điểm số (collaborative/content/activity)
        :param score_components: Dict chứa điểm số từ các thành phần khác nhau
        :return: Chuỗi mô tả lý do đề xuất
        """
        # Trích xuất điểm gốc (trước khi nhân trọng số) để đánh giá chính xác hơn
        collaborative_score = score_components.get('collaborative', 0) / self.weights['collaborative'] if self.weights['collaborative'] > 0 else 0
        content_score = score_components.get('content', 0) / self.weights['content_based'] if self.weights['content_based'] > 0 else 0
        activity_score = score_components.get('activity', 0) / self.weights['activity_signal'] if self.weights['activity_signal'] > 0 else 0
        
        # Tạo lý do chi tiết hơn dựa trên nguồn chính và điểm thực tế
        if primary_source == 'collaborative':
            if collaborative_score > 0.8:
                return "Rất phù hợp dựa trên sở thích của những người dùng tương tự bạn"
            elif collaborative_score > 0.6:
                return "Phù hợp với sở thích của người dùng tương tự bạn"
            else:
                return "Được đề xuất bởi người dùng có sở thích tương tự"
                
        elif primary_source == 'content':
            # Kiểm tra xem có metadata của công việc để tạo lý do cụ thể hơn không
            if content_score > 0.8:
                return "Rất phù hợp với hồ sơ kỹ năng và sở thích của bạn"
            elif content_score > 0.6:
                return "Phù hợp với hồ sơ kỹ năng và sở thích của bạn"
            else:
                return "Có thể phù hợp với hồ sơ kỹ năng của bạn"
                
        elif primary_source == 'activity':
            if self.activity_logs_df is not None and not self.activity_logs_df.empty and 'action_type' in self.activity_logs_df.columns:
                # Kiểm tra xem người dùng có ứng tuyển công việc nào không
                user_applied = self.activity_logs_df[
                    (self.activity_logs_df['user_id'] == self.user_id) & 
                    (self.activity_logs_df['action_type'] == 'APPLY')
                ]
                
                if not user_applied.empty:
                    return "Tương tự với các công việc bạn đã ứng tuyển gần đây"
                else:
                    # Kiểm tra nếu người dùng có click vào công việc nào không
                    user_clicked = self.activity_logs_df[
                        (self.activity_logs_df['user_id'] == self.user_id) & 
                        (self.activity_logs_df['action_type'] == 'CLICK')
                    ]
                    
                    if not user_clicked.empty:
                        return "Dựa trên các công việc bạn đã quan tâm gần đây"
                    else:
                        return "Dựa trên các công việc bạn đã xem gần đây"
            else:
                return "Phù hợp với hoạt động gần đây của bạn"
        else:
            return "Công việc được nhiều người quan tâm"

    def recommend(self, user_id, top_n=10, include_reasons=True):
        """
        Gợi ý công việc cho người dùng với lý do giải thích
        
        :param user_id: ID người dùng
        :param top_n: Số lượng gợi ý muốn trả về
        :param include_reasons: Có bao gồm lý do gợi ý hay không
        :return: Danh sách các gợi ý kèm lý do giải thích
        """
        # Lưu user_id vào thuộc tính để có thể truy cập trong _get_recommendation_reason
        self.user_id = user_id
        
        user_has_ratings = len(self.ratings_df[self.ratings_df['user_id'] == user_id]) > 0
        user_has_activities = (self.activity_logs_df is not None and 
                    not self.activity_logs_df.empty and 
                    'user_id' in self.activity_logs_df.columns and
                    len(self.activity_logs_df[self.activity_logs_df['user_id'] == user_id]) > 0)
        
        # Lấy danh sách các công việc đã được đánh giá bởi người dùng
        rated_jobs = set()
        if user_has_ratings:
            rated_jobs = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['job_id'])
        
        # Lấy danh sách các công việc đã được xem bởi người dùng
        viewed_jobs = set()
        if user_has_activities and 'action_type' in self.activity_logs_df.columns:
            # Lấy tất cả các công việc mà người dùng đã có hoạt động (VIEW, CLICK, APPLY, etc.)
            viewed_jobs = set(self.activity_logs_df[
                (self.activity_logs_df['user_id'] == user_id)
            ]['job_id'])
        
        # Kết hợp các công việc đã xem và đã đánh giá
        excluded_jobs = rated_jobs.union(viewed_jobs)
        
        print(f"User {user_id} has rated {len(rated_jobs)} jobs")
        print(f"User {user_id} has viewed {len(viewed_jobs)} jobs")
        print(f"Excluding {len(excluded_jobs)} jobs from recommendations")
        
        # Nếu là người dùng mới, sử dụng đề xuất phổ biến nhưng loại bỏ các công việc đã xem/đánh giá
        if not user_has_ratings and not user_has_activities:
            popular_recs = self._popular_recommendations(top_n * 2)  # Lấy nhiều hơn để có dư sau khi lọc
            recommendations = []
            for rec in popular_recs:
                if rec['job_id'] not in excluded_jobs and len(recommendations) < top_n:
                    if include_reasons:
                        rec['reason'] = "Công việc được nhiều người quan tâm"
                        rec['reason_source'] = 'popular'
                    recommendations.append(rec)
            return recommendations
        
        # Thực hiện các loại recommendation với số lượng lớn hơn để đảm bảo đủ sau khi lọc
        expanded_n = top_n * 3  # Lấy gấp 3 lần để đảm bảo đủ kết quả sau khi lọc
        collaborative_recs = self._collaborative_recommendations(user_id, expanded_n)
        content_recs = self._content_based_recommendations(user_id, expanded_n)
        activity_recs = self._activity_signal_recommendations(user_id, expanded_n)
        
        # Kết hợp điểm recommendation, chỉ bao gồm các công việc chưa xem/đánh giá
        combined_scores = {}
        source_scores = {}  # Theo dõi điểm số từ các nguồn khác nhau
        
        for rec in collaborative_recs:
            job_id = rec['job_id']
            if job_id not in excluded_jobs:
                score = rec.get('score', 0) * self.weights['collaborative']
                combined_scores[job_id] = combined_scores.get(job_id, 0) + score
                source_scores.setdefault(job_id, {})['collaborative'] = score
        
        for rec in content_recs:
            job_id = rec['job_id']
            if job_id not in excluded_jobs:
                score = rec.get('score', 0) * self.weights['content_based']
                combined_scores[job_id] = combined_scores.get(job_id, 0) + score
                source_scores.setdefault(job_id, {})['content'] = score
        
        for rec in activity_recs:
            job_id = rec['job_id']
            if job_id not in excluded_jobs:
                score = rec.get('score', 0) * self.weights['activity_signal']
                combined_scores[job_id] = combined_scores.get(job_id, 0) + score
                source_scores.setdefault(job_id, {})['activity'] = score
        
        # Xác định nguồn gợi ý chính cho mỗi công việc
        primary_sources = {}
        for job_id, scores in source_scores.items():
            # Tìm nguồn đóng góp nhiều nhất cho gợi ý này
            if scores:
                primary_source = max(scores.items(), key=lambda x: x[1])[0]
                primary_sources[job_id] = primary_source
            else:
                primary_sources[job_id] = 'popular'
        
        # Sắp xếp và trả về top N recommendation
        final_recommendations = []
        for job_id, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            job_data = self.jobs_df[self.jobs_df['job_id'] == job_id]
            if not job_data.empty:
                job_details = {
                    'job_id': job_id, 
                    'score': score,
                    'is_new': True  # Đánh dấu đây là công việc mới đối với người dùng
                }
                
                # Thêm các thông tin chi tiết của công việc nếu có
                for col in job_data.columns:
                    if col != 'job_id':
                        job_details[col] = job_data[col].values[0]
                
                # Thêm lý do gợi ý nếu được yêu cầu
                if include_reasons:
                    primary_source = primary_sources.get(job_id, 'popular')
                    job_details['reason'] = self._get_recommendation_reason(
                        primary_source, 
                        source_scores.get(job_id, {})
                    )
                    job_details['reason_source'] = primary_source
                    
                final_recommendations.append(job_details)
        
        # Nếu không đủ kết quả sau khi lọc, có thể bổ sung bằng các công việc phổ biến mới
        if len(final_recommendations) < top_n:
            needed = top_n - len(final_recommendations)
            popular_recs = self._popular_recommendations(needed * 2)
            existing_job_ids = {rec['job_id'] for rec in final_recommendations}
            
            for rec in popular_recs:
                job_id = rec['job_id']
                if (job_id not in excluded_jobs and 
                    job_id not in existing_job_ids and 
                    len(final_recommendations) < top_n):
                    rec['is_new'] = True
                    if include_reasons:
                        rec['reason'] = "Công việc được nhiều người quan tâm"
                        rec['reason_source'] = 'popular'
                    final_recommendations.append(rec)
        
        return final_recommendations


def create_hybrid_recommender(
    jobs_df, 
    ratings_df, 
    activity_logs_df=None,
    content_weight=0.3,
    collaborative_weight=0.4,
    activity_weight=0.3,
    min_ratings=2
):
    """
    Hàm tiện ích để tạo recommender
    
    :param jobs_df: DataFrame chứa thông tin công việc
    :param ratings_df: DataFrame chứa đánh giá
    :param activity_logs_df: DataFrame chứa nhật ký hoạt động
    :param content_weight: Trọng số cho content-based filtering
    :param collaborative_weight: Trọng số cho collaborative filtering
    :param activity_weight: Trọng số cho activity-based filtering
    :param min_ratings: Số lượng đánh giá tối thiểu để xem xét
    :return: HybridRecommender instance
    """
    if activity_logs_df is None or activity_logs_df.empty:
        activity_logs_df = pd.DataFrame(columns=['user_id', 'job_id', 'action_type'])
    return HybridRecommender(
        jobs_df, 
        ratings_df, 
        activity_logs_df,
        content_weight=content_weight,
        collaborative_weight=collaborative_weight,
        activity_weight=activity_weight
    )