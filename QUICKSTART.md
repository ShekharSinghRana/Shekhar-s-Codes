# Quick Start Guide

## Run the Project in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python train_model.py
```

Expected output:
- Dataset loaded: 98 rows
- Model accuracy: ~95%
- Model saved to `models/student_risk_model.joblib`

### Step 3: Launch the Web App
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## Test the App

Try these example scenarios:

### Example 1: High-Risk Student
- Study hours: 3
- Attendance: 55%
- Past failures: 3
- Parental support: low
- Extra classes: no
- Health: 2

Expected: **AT RISK** with recommendations to improve

### Example 2: Low-Risk Student
- Study hours: 12
- Attendance: 90%
- Past failures: 0
- Parental support: high
- Extra classes: yes
- Health: 5

Expected: **NOT AT RISK** with encouragement to continue

## Troubleshooting

### Model not found error
If you see "Trained model not found", make sure you ran `python train_model.py` first.

### Import errors
Make sure all dependencies are installed: `pip install -r requirements.txt`

### Streamlit not found
Add Python Scripts to PATH or use: `python -m streamlit run app.py`

## Project Demo Tips

1. **Show the dataset**: Open `data/students.csv` in Excel to show the training data
2. **Run training**: Execute `train_model.py` and explain the metrics
3. **Demo the app**: Test with different student profiles
4. **Explain recommendations**: Show how the system provides personalized advice
5. **Discuss real-world impact**: How this helps identify at-risk students early

## For Your Report/Presentation

Key points to mention:
- **Problem**: Early identification of at-risk students
- **Solution**: ML-based prediction with actionable recommendations
- **Dataset**: 98 student records with 6 features
- **Model**: Random Forest Classifier (95% accuracy)
- **Impact**: Helps educators intervene early and provide targeted support
- **Technology**: Python, scikit-learn, Streamlit
- **Alignment**: Covers Samsung Innovation Campus chapters 3-6, 10-11
