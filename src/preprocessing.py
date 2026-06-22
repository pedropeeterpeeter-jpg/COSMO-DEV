import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    TARGET_COLUMN,
    SCALER_PATH
)


def preprocess_data(df):
    """
    Pré-processa o dataset completo para treino do modelo.

    Passos:
        1. Remove colunas irrelevantes ou com data leakage
        2. Remove linhas com valores em falta
        3. Separa features (X) da variável alvo (y)
        4. Aplica OneHotEncoder nas colunas categóricas
        5. Aplica StandardScaler nas colunas numéricas
        6. Guarda o preprocessor em disco para uso em predict.py

    Devolve:
        X_processed : numpy array com os dados transformados
        y           : Series com os rótulos (Estudando / Abandonou)
        feature_names: lista com os nomes das colunas após transformação
    """

    # ─────────────────────────────────────────────────────────────────
    # Remoção de colunas:
    # - "ID": identificador sem valor preditivo
    # - "Nome": dado pessoal sem influência no abandono
    # - "Principal_Causa_Abandono": data leakage — esta informação só
    #   existe DEPOIS do abandono acontecer; incluí-la ensinaria o
    #   modelo a "fazer batota" em vez de generalizar
    # ─────────────────────────────────────────────────────────────────
    colunas_remover = ["ID", "Nome", "Principal_Causa_Abandono"]
    colunas_existentes = [c for c in colunas_remover if c in df.columns]
    df = df.drop(columns=colunas_existentes)

    df = df.dropna()

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

    # ─────────────────────────────────────────────────────────────────
    # NOTA sobre handle_unknown="ignore":
    # Se em produção chegar uma província ou município que não existia
    # no treino, o OneHotEncoder não vai crashar — vai simplesmente
    # ignorar esse valor (todas as colunas one-hot ficam a zero).
    # É o comportamento mais robusto para este contexto.
    # ─────────────────────────────────────────────────────────────────
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols
            ),
            (
                "num",
                StandardScaler(),
                numeric_cols
            )
        ]
    )

    # fit_transform aqui: aprende os encoders e escalas com os dados históricos
    X_processed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out().tolist()

    # Guarda o preprocessor para ser reutilizado em predict.py
    # (apenas .transform(), nunca .fit_transform() em dados novos)
    joblib.dump(preprocessor, SCALER_PATH)

    print(f"Preprocessor guardado em: {SCALER_PATH}")
    print(f"Shape dos dados processados: {X_processed.shape}")
    print(f"Número de features: {len(feature_names)}")

    return X_processed, y, feature_names
