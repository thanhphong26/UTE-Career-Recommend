import pandas as pd
import numpy as np
from datetime import datetime, timezone
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional, Dict


class HybridRecommender:
    def __init__(self, jobs_df, ratings_df, activity_logs_df=None):
        """
        Khởi tạo recommender với nhiều kỹ thuật
        
        :param jobs_df: DataFrame chứa thông tin công việc
        :param ratings_df: DataFrame chứa đánh giá
        :param activity_logs_df: DataFrame chứa nhật ký hoạt động
        """
        self.jobs_df = jobs_df
        self.ratings_df = ratings_df
        self.activity_logs_df = activity_logs_df
        
        # Xây dựng ma trận đánh giá để collaborative filtering
        self._build_rating_matrix()
        
        # Xây dựng vector đặc trưng cho content-based filtering
        self._build_content_vectors()
        
        # Trọng số cho từng phương pháp recommendation
        self.weights = {
            'collaborative': 0.4,
            'content_based': 0.3,
            'activity_signal': 0.3
        }
    
    def _build_rating_matrix(self):
        """
        Xây dựng ma trận đánh giá để collaborative filtering
        """
        user_job_matrix = self.ratings_df.pivot_table(
            index='user_id', 
            columns='job_id', 
            fill_value=0
        )
        
        self.rating_matrix = csr_matrix(user_job_matrix.values)
        self.user_ids = user_job_matrix.index
        self.job_ids = user_job_matrix.columns
    
    def _build_content_vectors(self):
        """
        Xây dựng vector đặc trưng cho content-based filtering
        """
        # Kết hợp tiêu đề và mô tả để tạo vector
        self.jobs_df['content'] = self.jobs_df['job_title'] + ' ' + self.jobs_df['job_description']
        
        # Sử dụng TF-IDF để vector hóa nội dung
        tfidf = TfidfVectorizer(stop_words='english')
        self.content_matrix = tfidf.fit_transform(self.jobs_df['content'])
    
    def _collaborative_recommendations(self, user_id, top_n):
        """
        Recommendation dựa trên hành vi người dùng tương tự
        
        :param user_id: ID người dùng
        :param top_n: Số lượng recommendation
        :return: Danh sách recommendation
        """
        try:
            user_index = np.where(self.user_ids == user_id)[0]
            if len(user_index) == 0:
                return []
            
            user_index = user_index[0]
            user_ratings = self.rating_matrix[user_index]
            
            # Tính điểm tương đồng với các công việc đã được đánh giá
            sim_scores = cosine_similarity(user_ratings, self.rating_matrix).flatten()
            similar_users_indices = sim_scores.argsort()[::-1][1:6]  # Top 5 người dùng tương tự
            
            recommendations = {}
            for idx in similar_users_indices:
                user_job_ratings = self.ratings_df[
                    self.ratings_df['user_id'] == self.user_ids[idx]
                ]
                for _, row in user_job_ratings.iterrows():
                    job_id = row['job_id']
                    if job_id not in self.ratings_df[
                        self.ratings_df['user_id'] == user_id
                    ]['job_id'].values:
                        recommendations[job_id] = recommendations.get(
                            job_id, 0
                        ) + row['rating'] * sim_scores[idx]
            
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
            # Lấy các công việc đã được đánh giá bởi người dùng
            user_rated_jobs = self.ratings_df[
                self.ratings_df['user_id'] == user_id
            ]['job_id']
            print(f"User {user_id} rated jobs: {user_rated_jobs.tolist()}")
            
            if len(user_rated_jobs) == 0:
                return []
            
            # Tính độ tương đồng nội dung
            user_profile = np.asarray(self.content_matrix[self.jobs_df['job_id'].isin(user_rated_jobs)].mean(axis=0)).reshape(1, -1)

            content_sim_scores = cosine_similarity(user_profile, self.content_matrix).flatten()

            # Loại bỏ các công việc đã đánh giá
            mask = ~self.jobs_df['job_id'].isin(user_rated_jobs)
                
            recommendations = [
                {
                    'job_id': job_id, 
                    'score': score
                } 
                for job_id, score in zip(
                    self.jobs_df.loc[mask, 'job_id'], 
                    content_sim_scores[mask]
                )
            ]
            
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
        if self.activity_logs_df is None or self.activity_logs_df.empty:
            return []
        
        try:
            # Phân tích nhật ký hoạt động của người dùng
            user_logs = self.activity_logs_df[
                self.activity_logs_df['user_id'] == user_id
            ]
            
            activity_scores = user_logs.groupby('job_id').size()
            
            recommendations = [
                {
                    'job_id': job_id, 
                    'score': score
                } 
                for job_id, score in activity_scores.items()
            ]
            
            return sorted(
                recommendations, 
                key=lambda x: x['score'], 
                reverse=True
            )[:top_n]
        except Exception as e:
            print(f"Activity signal filtering error: {e}")
            return []
    
    def recommend(self, user_id, top_n=5):
        """
        Kết hợp các phương pháp recommendation
        
        :param user_id: ID người dùng
        :param top_n: Số lượng recommendation
        :return: Danh sách recommendation
        """
        collaborative_recs = self._collaborative_recommendations(user_id, top_n)
        content_recs = self._content_based_recommendations(user_id, top_n)
        activity_recs = self._activity_signal_recommendations(user_id, top_n)
        
        # Kết hợp điểm recommendation
        combined_scores: Dict[int, float] = {}
        
        for rec_list, weight in [
            (collaborative_recs, self.weights['collaborative']),
            (content_recs, self.weights['content_based']),
            (activity_recs, self.weights['activity_signal'])
        ]:
            for rec in rec_list:
                job_id = rec['job_id']
                score = rec.get('score', 1.0)
                combined_scores[job_id] = combined_scores.get(job_id, 0) + score * weight
        
        # Sắp xếp và trả về top N recommendation
        final_recommendations = sorted(
            [
                {
                    'job_id': job_id, 
                    'score': score,
                    'job_title': self.jobs_df[
                        self.jobs_df['job_id'] == job_id
                    ]['job_title'].values[0]
                } 
                for job_id, score in combined_scores.items()
            ],
            key=lambda x: x['score'], 
            reverse=True
        )[:top_n]
        
        return final_recommendations

def create_hybrid_recommender(jobs_df, ratings_df, activity_logs_df=None):
    """
    Hàm tiện ích để tạo recommender
    
    :param jobs_df: DataFrame chứa thông tin công việc
    :param ratings_df: DataFrame chứa đánh giá
    :param activity_logs_df: DataFrame chứa nhật ký hoạt động
    :return: HybridRecommender instance
    """
    return HybridRecommender(jobs_df, ratings_df, activity_logs_df)