import pandas as pd
import numpy as np
from random import choices, randint, uniform

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters and their distributions
n_records = 10000

data = {
    # Demographics
    'age': np.random.randint(18, 80, n_records),
    'gender': choices(['Male', 'Female', 'Non-binary'], weights=[0.48, 0.48, 0.04], k=n_records),
    'ethnicity': choices(['Caucasian', 'African American', 'Asian', 'Hispanic', 'Other'], 
                        weights=[0.6, 0.13, 0.05, 0.18, 0.04], k=n_records),
    'education': choices(['High School', 'Some College', "Bachelor's", "Master's", "PhD"], 
                         weights=[0.3, 0.2, 0.3, 0.15, 0.05], k=n_records),
    'employment': choices(['Employed', 'Unemployed', 'Student', 'Retired'], 
                         weights=[0.6, 0.1, 0.2, 0.1], k=n_records),
    'income': np.random.randint(20000, 150000, n_records),
    
    # Physical Health
    'bmi': np.round(np.random.normal(26, 5, n_records), 1),
    'sleep_quality': np.random.randint(3, 9, n_records),
    'exercise_frequency': np.random.randint(0, 7, n_records),
    'chronic_conditions': np.random.randint(0, 5, n_records),
    
    # Mental Health Scores
    'phq9_score': np.random.randint(0, 27, n_records),
    'gad7_score': np.random.randint(0, 21, n_records),
    'stress_score': np.random.randint(0, 40, n_records),
    'resilience_score': np.random.randint(3, 30, n_records),
    
    # Social/Lifestyle
    'social_support': np.random.randint(1, 10, n_records),
    'loneliness_score': np.random.randint(1, 10, n_records),
    'alcohol_consumption': np.random.randint(0, 7, n_records),
    'screen_time': np.round(np.random.uniform(2, 12, n_records), 1),
    'caffeine_intake': np.random.randint(0, 6, n_records),
    
    # Work/Environment
    'work_balance': np.random.randint(1, 5, n_records),
    'neighborhood_safety': np.random.randint(1, 5, n_records),
    
    # Target variable
    'mental_health_status': None
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate target variable based on weighted scores
def determine_mental_health(row):
    weighted_score = (row['phq9_score'] * 0.35 + 
                     row['gad7_score'] * 0.25 + 
                     row['stress_score'] * 0.15 +
                     row['loneliness_score'] * 0.1 -
                     row['social_support'] * 0.08 - 
                     row['exercise_frequency'] * 0.05 -
                     row['resilience_score'] * 0.02)
    
    if weighted_score < 5:
        return 'Excellent'
    elif weighted_score < 10:
        return 'Good'
    elif weighted_score < 15:
        return 'poor'
    else:
        return 'Poor'

df['mental_health_status'] = df.apply(determine_mental_health, axis=1)

# Save to CSV with the specified name
df.to_csv('mentaldata.csv', index=False)
print("Dataset 'mentaldata.csv' with 10,000 records created successfully!")