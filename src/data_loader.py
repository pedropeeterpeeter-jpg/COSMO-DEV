import pandas as pd
from src.config import RAW_DATA


def load_data():
    """
    Lê o dataset original em formato .xlsx e devolve um DataFrame.

    CORRECÇÃO: adicionada verificação de existência do ficheiro antes
    de tentar abrir. Sem isto, o pandas lança um erro genérico que
    não indica claramente que o problema é o caminho do ficheiro.
    """

    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"\n[ERRO] Dataset não encontrado em:\n  {RAW_DATA}\n"
            f"Verifica se o ficheiro .xlsx está na pasta data/raw/"
        )

    print(f"A carregar dataset de: {RAW_DATA}")

    df = pd.read_excel(RAW_DATA, engine="openpyxl")

    print(f"Dataset carregado: {df.shape[0]} registos, {df.shape[1]} colunas")

    return df
