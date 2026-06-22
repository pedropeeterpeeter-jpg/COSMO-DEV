import joblib

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from data_loader import load_data
from preprocessing import preprocess_data

from src.config import  (
    MODEL_PATH,
    TEST_SIZE,
    RANDOM_STATE
)

def train():

    df = load_data()

    X, y, _ = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = SVC(
        kernel="rbf",
        probability=True,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    acc = accuracy_score(
        y_test,
        predictions
    )

    print("\nAccuracy:")
    print(acc)

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nModelo salvo!")

if __name__ == "__main__":
    train()