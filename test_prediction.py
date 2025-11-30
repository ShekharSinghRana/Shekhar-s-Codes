"""
Simple script to test the trained model with sample predictions
Run this after training the model to verify it works correctly
"""
import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "student_risk_model.joblib")

def test_predictions():
    # Load the trained model
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    print("✓ Model loaded successfully\n")
    
    # Test Case 1: High-risk student
    print("=" * 60)
    print("TEST CASE 1: High-Risk Student Profile")
    print("=" * 60)
    student1 = pd.DataFrame({
        "study_hours_per_week": [3],
        "attendance_rate": [55],
        "past_failures": [3],
        "parental_support": ["low"],
        "extra_classes": ["no"],
        "health": [2]
    })
    
    pred1 = model.predict(student1)[0]
    proba1 = model.predict_proba(student1)[0]
    
    print("Input:")
    print(f"  - Study hours/week: 3")
    print(f"  - Attendance: 55%")
    print(f"  - Past failures: 3")
    print(f"  - Parental support: low")
    print(f"  - Extra classes: no")
    print(f"  - Health: 2/5")
    print(f"\nPrediction: {'AT RISK ⚠️' if pred1 == 1 else 'SAFE ✓'}")
    print(f"Risk probability: {proba1[1]:.2%}")
    print(f"Safe probability: {proba1[0]:.2%}\n")
    
    # Test Case 2: Low-risk student
    print("=" * 60)
    print("TEST CASE 2: Low-Risk Student Profile")
    print("=" * 60)
    student2 = pd.DataFrame({
        "study_hours_per_week": [12],
        "attendance_rate": [90],
        "past_failures": [0],
        "parental_support": ["high"],
        "extra_classes": ["yes"],
        "health": [5]
    })
    
    pred2 = model.predict(student2)[0]
    proba2 = model.predict_proba(student2)[0]
    
    print("Input:")
    print(f"  - Study hours/week: 12")
    print(f"  - Attendance: 90%")
    print(f"  - Past failures: 0")
    print(f"  - Parental support: high")
    print(f"  - Extra classes: yes")
    print(f"  - Health: 5/5")
    print(f"\nPrediction: {'AT RISK ⚠️' if pred2 == 1 else 'SAFE ✓'}")
    print(f"Risk probability: {proba2[1]:.2%}")
    print(f"Safe probability: {proba2[0]:.2%}\n")
    
    # Test Case 3: Borderline student
    print("=" * 60)
    print("TEST CASE 3: Borderline Student Profile")
    print("=" * 60)
    student3 = pd.DataFrame({
        "study_hours_per_week": [6],
        "attendance_rate": [72],
        "past_failures": [1],
        "parental_support": ["medium"],
        "extra_classes": ["no"],
        "health": [4]
    })
    
    pred3 = model.predict(student3)[0]
    proba3 = model.predict_proba(student3)[0]
    
    print("Input:")
    print(f"  - Study hours/week: 6")
    print(f"  - Attendance: 72%")
    print(f"  - Past failures: 1")
    print(f"  - Parental support: medium")
    print(f"  - Extra classes: no")
    print(f"  - Health: 4/5")
    print(f"\nPrediction: {'AT RISK ⚠️' if pred3 == 1 else 'SAFE ✓'}")
    print(f"Risk probability: {proba3[1]:.2%}")
    print(f"Safe probability: {proba3[0]:.2%}\n")
    
    print("=" * 60)
    print("All tests completed successfully! ✓")
    print("=" * 60)

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run 'python train_model.py' first.")
    else:
        test_predictions()
