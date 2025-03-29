import pandas as pd

def process_user_activity(user_activity):
    """
    Process user activity data
    
    :param user_activity: List of user activity dictionaries
    :return: Processed DataFrame of user activities
    """
    processed_data = []
    for activity in user_activity:
        processed_data.append({
            "user_id": activity["user_id"],
            "job_id": activity["job_id"],
            "rating": activity.get("rating", 0)
        })
    return pd.DataFrame(processed_data)

def extract_job_features(job_list):
    """
    Extract features from job descriptions
    
    :param job_list: List of job dictionaries
    :return: DataFrame with job features
    """
    job_df = pd.DataFrame(job_list)
    
    # You can add more feature extraction logic here
    # For now, just returning the DataFrame
    return job_df