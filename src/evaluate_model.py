import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from sklearn.model_selection import train_test_split

from data_loader import load_data
from preprocessing import preprocess_data

from config import (
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE
)


def evaluate():

    df = load_data()

    X, y, _ = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n===== MÉTRICAS =====")

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("ROC AUC  :", roc_auc_score(y_test, y_prob))

    cm = confusion_matrix(y_test, y_pred)

    ConfusionMatrixDisplay(cm).plot()

    plt.savefig(
        "../imagens/matriz_confusao.png",
        bbox_inches="tight"
    )

    RocCurveDisplay.from_predictions(
        y_test,
        y_prob
    )

    plt.savefig(
        "../imagens/curva_roc.png",
        bbox_inches="tight"
    )


if __name__ == "__main__":

    evaluate()