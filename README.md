# AI-powered-Resume-Screening-and-Ranking-System
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset (resumes and job descriptions)
data = {
    "resume_text": [
        "Experienced software engineer with expertise in Python and machine learning.",
        "Data scientist skilled in data analysis, Python, and TensorFlow.",
        "Web developer with experience in JavaScript, React, and Node.js.",
        "Machine learning engineer with a background in deep learning and NLP.",
        "Frontend developer specializing in HTML, CSS, and JavaScript.",
    ],
    "job_description": [
        "We are looking for a software engineer with Python and machine learning skills.",
        "Seeking a data scientist proficient in Python and data analysis.",
        "Hiring a web developer with expertise in JavaScript and React.",
        "Looking for a machine learning engineer with NLP experience.",
        "Need a frontend developer skilled in HTML, CSS, and JavaScript.",
    ],
    "label": [1, 1, 0, 1, 0],  # 1 = Match, 0 = No Match
}

# Convert dataset to a DataFrame
df = pd.DataFrame(data)

# Step 1: Preprocess the data
# Combine resume text and job description for feature extraction
df["combined_text"] = df["resume_text"] + " " + df["job_description"]

# Step 2: Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["combined_text"])
y = df["label"]

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a machine learning model (Random Forest Classifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 6: Resume Ranking
# Simulate new resumes and job descriptions for ranking
new_resumes = [
    "Python developer with experience in Django and Flask.",
    "Data analyst skilled in SQL and Excel.",
    "Full-stack developer with knowledge of React and Node.js.",
]
job_description = "We need a Python developer with Django experience."

# Combine new resumes with the job description
new_data = [job_description + " " + resume for resume in new_resumes]

# Transform new data using the same vectorizer
X_new = vectorizer.transform(new_data)

# Predict the match score for each resume
match_scores = model.predict_proba(X_new)[:, 1]  # Probability of being a match

# Rank resumes based on match scores
ranked_resumes = sorted(zip(new_resumes, match_scores), key=lambda x: x[1], reverse=True)

# Display ranked resumes
print("\nRanked Resumes:")
for idx, (resume, score) in enumerate(ranked_resumes, 1):
    print(f"{idx}. Resume: {resume} | Match Score: {score:.4f}")
