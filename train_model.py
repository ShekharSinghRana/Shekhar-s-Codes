import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Paths
DATA_PATH = os.path.join("data", "students.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "student_risk_model.joblib")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def create_labels(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """
    Create a binary 'at_risk' label:
    1 = at risk (final_score < threshold)
    0 = not at risk
    """
    if "final_score" not in df.columns:
        raise KeyError("Column 'final_score' not found in dataset.")
    
    df = df.copy()
    df["at_risk"] = (df["final_score"] < threshold).astype(int)
    return df


def build_pipeline(df: pd.DataFrame, target_col: str = "at_risk") -> Pipeline:
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    
    if not numeric_cols and not categorical_cols:
        raise ValueError("No features detected. Check your dataset.")
    
    # Preprocessing for numeric data
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    
    # Model
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    
    # Full pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Train
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluation on test set:")
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))
    
    return model


def main():
    # Load data
    df = load_data(DATA_PATH)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Create 'at_risk' label
    df = create_labels(df, threshold=50.0)
    print("Created 'at_risk' label based on final_score < 50.")
    
    # Drop final_score from features; we only use it to create label
    df_model = df.drop(columns=["final_score"])
    
    # Train model
    model = build_pipeline(df_model, target_col="at_risk")
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
