import os
import joblib
import streamlit as st
import pandas as pd

MODEL_PATH = os.path.join("models", "student_risk_model.joblib")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            f"Run 'python train_model.py' first."
        )
    model = joblib.load(MODEL_PATH)
    return model


def build_input_dataframe(
    study_hours_per_week: float,
    attendance_rate: float,
    past_failures: int,
    parental_support: str,
    extra_classes: str,
    health: int
) -> pd.DataFrame:
    """
    Build a one-row DataFrame matching the training features.
    Note: final_score is NOT included here because it's the target in training.
    """
    data = {
        "study_hours_per_week": [study_hours_per_week],
        "attendance_rate": [attendance_rate],
        "past_failures": [past_failures],
        "parental_support": [parental_support],
        "extra_classes": [extra_classes],
        "health": [health],
    }
    return pd.DataFrame(data)


def generate_recommendations(row: pd.Series) -> str:
    recs = []
    
    if row["study_hours_per_week"] < 8:
        recs.append(
            "- Increase weekly study hours. Aim for at least 8–12 hours with a fixed timetable."
        )
    
    if row["attendance_rate"] < 75:
        recs.append(
            "- Improve class attendance. Try to maintain at least 80–85% attendance."
        )
    
    if row["past_failures"] >= 2:
        recs.append(
            "- Focus on weak subjects and ask for remedial classes or extra help from teachers."
        )
    
    if row["parental_support"] == "low":
        recs.append(
            "- Involve parents/guardians in your study plan, even with simple weekly progress discussions."
        )
    
    if row["extra_classes"] == "no":
        recs.append(
            "- Consider joining extra classes, online lectures, or peer study groups."
        )
    
    if row["health"] <= 2:
        recs.append(
            "- Take care of your health: sleep properly, eat on time, and manage stress."
        )
    
    if not recs:
        recs.append(
            "- Continue your current habits, but revise regularly and attempt previous year question papers."
        )
    
    return "\n".join(recs)


def main():
    st.title("🎓 Student Performance Risk Prediction")
    st.write(
        """
        This tool predicts whether a student is **academically at risk** 
        based on study habits, attendance, and support factors, 
        and suggests personalised improvement tips.
        """
    )
    
    model = load_model()
    
    st.subheader("Enter student details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        study_hours = st.number_input(
            "Study hours per week",
            min_value=0.0,
            max_value=80.0,
            value=6.0,
            step=0.5
        )
        attendance = st.slider(
            "Attendance rate (%)",
            min_value=0,
            max_value=100,
            value=75,
            step=1
        )
        past_failures = st.number_input(
            "Number of past failures",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )
    
    with col2:
        parental_support = st.selectbox(
            "Parental support",
            options=["low", "medium", "high"],
            index=1
        )
        extra_classes = st.radio(
            "Takes extra classes / coaching?",
            options=["yes", "no"],
            index=0,
            horizontal=True
        )
        health = st.slider(
            "Health status (1 = very bad, 5 = excellent)",
            min_value=1,
            max_value=5,
            value=4,
            step=1
        )
    
    if st.button("Predict"):
        # Build input
        input_df = build_input_dataframe(
            study_hours_per_week=study_hours,
            attendance_rate=attendance,
            past_failures=past_failures,
            parental_support=parental_support,
            extra_classes=extra_classes,
            health=health
        )
        
        # Predict
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        
        at_risk_prob = proba[1]  # probability of class 1 (at risk)
        safe_prob = proba[0]
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        if pred == 1:
            st.error(
                f"⚠️ The model predicts this student is **AT RISK** "
                f"(Risk probability: {at_risk_prob:.2f})"
            )
        else:
            st.success(
                f"✅ The model predicts this student is **NOT AT RISK** "
                f"(Safe probability: {safe_prob:.2f})"
            )
        
        # Recommendations
        st.subheader("Suggested Actions")
        rec_text = generate_recommendations(input_df.iloc[0])
        st.write(rec_text)


if __name__ == "__main__":
    main()
