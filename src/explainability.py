import shap
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.config import (
    RANDOM_FOREST_PATH,
    IMAGE_DIR
)

# Mapa de nomes técnicos → nomes legíveis para gestores escolares
NOMES_LEGÍVEIS = {
    "Idade":                "Idade",
    "Sexo":                 "Sexo",
    "Provincia":            "Província",
    "Municipio":            "Município",
    "Zona":                 "Zona",
    "Tipo_Instituicao":     "Tipo de Instituição",
    "Nivel_Ensino":         "Nível de Ensino",
    "Classe":               "Classe",
    "Renda_Familiar":       "Renda Familiar",
    "Numero_Irmaos":        "Número de Irmãos",
    "Distancia_Escola_km":  "Distância da Escola",
    "Meio_Transporte":      "Meio de Transporte",
    "Trabalha":             "Trabalha",
    "Repetiu_Ano":          "Repetiu Ano",
    "Numero_Reprovacoes":   "Nº de Reprovações",
    "Apoio_Familiar":       "Apoio Familiar",
    "Bullying":             "Bullying",
    "Acesso_Internet":      "Acesso à Internet",
    "Pretende_Retornar":    "Pretende Retornar",
}


def limpar_nome(nome_tecnico):
    """
    Converte nomes técnicos do OneHotEncoder para nomes legíveis.
    Exemplo: cat__Sexo_Masculino → Sexo
             num__Distancia_Escola_km → Distância da Escola
    """
    # Remove prefixos cat__ e num__
    nome = nome_tecnico.replace("cat__", "").replace("num__", "")
    # Pega apenas a parte antes do underscore de valor (ex: Sexo_Masculino → Sexo)
    base = nome.split("_")[0] if "_" in nome else nome
    # Tenta encontrar no mapa, senão devolve o nome limpo
    for chave, valor in NOMES_LEGÍVEIS.items():
        if chave.lower() in nome.lower():
            return valor
    return base


def generate_global_explanation():
    """
    Gera o gráfico SHAP Global em formato PIZZA.
    Mostra as top 8 variáveis mais influentes no modelo
    com percentagem relativa de importância.
    Guardado em static/imagens/shap_global.png
    """

    Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, y, feature_names = preprocess_data(df)

    if hasattr(X, "toarray"):
        X = X.toarray()

    model = joblib.load(RANDOM_FOREST_PATH)

    print("A calcular SHAP Global...")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Importância média absoluta por variável
    if isinstance(shap_values, list):
        importancia = np.abs(shap_values[1]).mean(axis=0)
    elif shap_values.ndim == 3:
        importancia = np.abs(shap_values[:, :, 1]).mean(axis=0)
    else:
        importancia = np.abs(shap_values).mean(axis=0)

    # Top 8 variáveis mais importantes
    top_n = 8
    indices_top = np.argsort(importancia)[::-1][:top_n]

    labels_tecnicos = [feature_names[i] for i in indices_top]
    valores         = importancia[indices_top]

    # Limpa os nomes para exibição
    labels_limpos = [limpar_nome(l) for l in labels_tecnicos]

    # Remove duplicados mantendo o de maior valor
    vistos  = {}
    for label, valor in zip(labels_limpos, valores):
        if label not in vistos or valor > vistos[label]:
            vistos[label] = valor
    labels_finais = list(vistos.keys())
    valores_finais = list(vistos.values())

    # ─────────────────────────────────────────────────────────────────
    # GRÁFICO DE PIZZA
    # Cada fatia representa a importância relativa de uma variável
    # no total das 8 mais influentes do modelo.
    # ─────────────────────────────────────────────────────────────────
    cores = [
        "#1B4F72", "#2E86AB", "#27AE60", "#F39C12",
        "#E74C3C", "#8E44AD", "#16A085", "#D35400"
    ][:len(labels_finais)]

    fig, ax = plt.subplots(figsize=(9, 7))

    wedges, texts, autotexts = ax.pie(
        valores_finais,
        labels=None,
        autopct="%1.1f%%",
        startangle=140,
        colors=cores,
        pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )

    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
        at.set_fontweight("bold")

    ax.legend(
        wedges,
        labels_finais,
        title="Variáveis",
        title_fontsize=10,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        frameon=True,
        framealpha=0.9
    )

    ax.set_title(
        "Variáveis com maior influência no abandono escolar\n(importância relativa — modelo geral)",
        fontsize=12,
        fontweight="bold",
        pad=20
    )

    plt.tight_layout()

    plt.savefig(
        Path(IMAGE_DIR) / "shap_global.png",
        bbox_inches="tight",
        dpi=130
    )
    plt.close()

    print("SHAP Global (pizza) guardado em:", Path(IMAGE_DIR) / "shap_global.png")


if __name__ == "__main__":
    generate_global_explanation()
