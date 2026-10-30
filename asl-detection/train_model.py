"""STEP 2 - Train the classifier on the landmarks you collected.

Run:  python train_model.py

Reads data/landmarks.csv, trains a Random Forest to map the 63 hand-landmark
numbers to a letter, prints how accurate it is, and saves the model to
model/asl_model.joblib for the live detector to use.
"""
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "landmarks.csv")
MODEL_PATH = os.path.join(HERE, "model", "asl_model.joblib")


def main():
    if not os.path.exists(DATA_PATH):
        print("No data yet. Run  python collect_data.py  first.")
        return

    df = pd.read_csv(DATA_PATH)
    if len(df) < 20:
        print(f"Only {len(df)} samples. Collect more before training.")
        return

    X = df.drop(columns=["label"]).values   # the 63 landmark numbers
    y = df["label"].values                   # the letter

    print("Samples per letter:")
    print(df["label"].value_counts().sort_index().to_string())
    print(f"\nTotal: {len(df)} samples across {df['label'].nunique()} letters\n")

    # Hold out 20% to honestly measure accuracy on data the model didn't train on.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("Accuracy report (on held-out test data):")
    print(classification_report(y_test, clf.predict(X_test), zero_division=0))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print("Next: run  python detect_asl.py")


if __name__ == "__main__":
    main()
