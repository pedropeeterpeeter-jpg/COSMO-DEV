import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    RANDOM_FOREST_PATH,
    MODEL_DIR
)


def train_random_forest():
    """
    Treina o modelo Random Forest e guarda-o em disco.

    Sequência:
        1. Carrega o dataset
        2. Pré-processa (encode + scale + guarda preprocessor.pkl)
        3. Divide em treino/teste (80/20, estratificado)
        4. Treina o RandomForestClassifier
        5. Avalia e imprime métricas no terminal
        6. Guarda o modelo em models/random_forest.pkl

    Deve ser executado UMA VEZ antes de arrancar o servidor Flask.
    Re-executar recria o modelo e o preprocessor do zero.
    """

    print("=" * 50)
    print("TREINO DO MODELO RANDOM FOREST")
    print("=" * 50)

    df = load_data()

    # preprocess_data faz fit_transform e guarda o preprocessor.pkl
    X, y, feature_names = preprocess_data(df)

    print(f"\nClasses encontradas: {y.unique().tolist()}")
    print(f"Distribuição:\n{y.value_counts().to_string()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y          # garante proporção igual de classes em treino e teste
    )

    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Teste:  {X_test.shape[0]} amostras\n")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=RANDOM_STATE,
        class_weight="balanced"   # compensa desequilíbrio entre classes
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    print(f"Accuracy: {acc:.4f} ({acc*100:.1f}%)\n")
    print(classification_report(y_test, pred))

    # ─────────────────────────────────────────────────────────────────
    # CORRECÇÃO: removida a importação de SHAP_GLOBAL que não era usada
    # aqui — a geração do SHAP global é responsabilidade de
    # explainability.py, não do treino do modelo.
    # ─────────────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, RANDOM_FOREST_PATH)

    print(f"Modelo guardado em: {RANDOM_FOREST_PATH}")
    print("\nPróximo passo: executar  python -m src.explainability")


if __name__ == "__main__":
    train_random_forest()
