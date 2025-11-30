# AI-Based Student Performance Risk Prediction & Study Recommendation System

## Project Overview
A machine learning model that predicts whether a student is at risk of low performance based on study habits, attendance, and support factors, and suggests what they should improve.

## Project Structure
```
student_risk_project/
├─ data/
│  └─ students.csv          # Training dataset
├─ models/
│  └─ student_risk_model.joblib  # Trained model (created after training)
├─ train_model.py           # Model training script
├─ app.py                   # Streamlit web application
├─ requirements.txt         # Python dependencies
└─ README.md               # This file
```

## Setup Instructions

### 1. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
```

Activate it:
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
Make sure `data/students.csv` exists, then run:
```bash
python train_model.py
```

You should see:
- Dataset shape printed
- Training progress
- Confusion matrix
- Classification report (precision, recall, F1, accuracy)
- Confirmation that model is saved

### 4. Run the Web Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use the Web App

1. Enter student details:
   - Study hours per week
   - Attendance rate (%)
   - Number of past failures
   - Parental support level
   - Whether they take extra classes
   - Health status (1-5 scale)

2. Click "Predict" button

3. View results:
   - Risk prediction (At Risk / Not At Risk)
   - Probability scores
   - Personalized recommendations

## Dataset Information

The dataset contains 100 student records with the following features:

- **study_hours_per_week**: Weekly study hours (numeric)
- **attendance_rate**: Class attendance percentage 0-100 (numeric)
- **past_failures**: Number of previous failures (numeric)
- **parental_support**: Level of parental support (low/medium/high)
- **extra_classes**: Whether student takes extra classes (yes/no)
- **health**: Health status 1-5 scale (numeric)
- **final_score**: Final exam score 0-100 (target variable)

The model creates a binary label:
- **at_risk = 1** if final_score < 50
- **at_risk = 0** if final_score >= 50

## Technical Details

### Machine Learning Pipeline
1. **Data Preprocessing**:
   - Numeric features: StandardScaler
   - Categorical features: OneHotEncoder

2. **Model**: RandomForestClassifier
   - 200 estimators
   - No max depth limit
   - Random state: 42

3. **Evaluation**: 80-20 train-test split with stratification

### Technologies Used
- Python 3.8+
- pandas & numpy (data manipulation)
- scikit-learn (machine learning)
- joblib (model serialization)
- streamlit (web interface)

## Connection to Samsung Innovation Campus Curriculum

This project demonstrates concepts from:

- **Chapter 3 (EDA)**: Data exploration, distributions, correlations
- **Chapter 4 (Probability & Statistics)**: Thresholding, distributions, basic statistics
- **Chapter 5 & 6 (ML 1 & 2)**: Supervised learning, classification, train/test split, evaluation metrics, Random Forest, pipelines
- **Chapter 10 (Starting an AI Project)**: CRISP-DM process, Design Thinking, problem definition
- **Chapter 11 (Capstone)**: Applying AI to education and student success

## Future Enhancements

1. Add more features (socioeconomic factors, previous grades, etc.)
2. Implement deep learning models (neural networks)
3. Add NLP analysis of student feedback
4. Create visualization dashboard for EDA
5. Deploy to cloud platform (Streamlit Cloud, Heroku, etc.)

## License
Educational project for Samsung Innovation Campus
