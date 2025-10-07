import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from src.utils.io import data_path, read_csv, write_csv


def train_rf():
    df = read_csv(data_path("processed", "features.csv"))
    y = df["outcome_tier"]
    X = df.drop(columns=["outcome_tier"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    print(report)
    pd.DataFrame(cm, index=sorted(y.unique()), columns=sorted(y.unique())).to_csv(
        data_path("processed", "rf_confusion_matrix.csv")
    )
    return clf


if __name__ == "__main__":
    train_rf()
