import joblib
import numpy as np
import pandas as pd

from src.config import RANDOM_FOREST_PATH, SCALER_PATH
from src.local_explainability import generate_local_explanation

model        = joblib.load(RANDOM_FOREST_PATH)
preprocessor = joblib.load(SCALER_PATH)

NOMES_LEGIVEIS = {
    "Idade":               "Idade",
    "Sexo":                "Sexo",
    "Provincia":           "Província",
    "Municipio":           "Município",
    "Zona":                "Zona",
    "Tipo_Instituicao":    "Tipo de Instituição",
    "Nivel_Ensino":        "Nível de Ensino",
    "Classe":              "Classe",
    "Renda_Familiar":      "Renda Familiar",
    "Numero_Irmaos":       "Número de Irmãos",
    "Distancia_Escola_km": "Distância da Escola",
    "Meio_Transporte":     "Meio de Transporte",
    "Trabalha":            "Trabalha",
    "Repetiu_Ano":         "Repetiu Ano",
    "Numero_Reprovacoes":  "Nº de Reprovações",
    "Apoio_Familiar":      "Apoio Familiar",
    "Bullying":            "Bullying",
    "Acesso_Internet":     "Acesso à Internet",
    "Pretende_Retornar":   "Pretende Retornar",
}

VARIAVEIS_FORMULARIO = list(NOMES_LEGIVEIS.keys())


def limpar_nome(nome_tecnico):
    nome = nome_tecnico.replace("cat__", "").replace("num__", "")
    for chave, valor in NOMES_LEGIVEIS.items():
        if chave.lower() in nome.lower():
            return valor
    return nome.split("_")[0] if "_" in nome else nome


def predict_student(student_data):
    df = pd.DataFrame([student_data])
    X  = preprocessor.transform(df)

    prediction    = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    classes        = model.classes_
    idx_abandonou  = list(classes).index("Abandonou")
    idx_estudando  = list(classes).index("Estudando")

    prob_abandonou = round(float(probabilities[idx_abandonou]) * 100, 1)
    prob_estudando = round(float(probabilities[idx_estudando]) * 100, 1)
    classe         = str(prediction)

    if prob_abandonou >= 75:
        risco = "Alto"
    elif prob_abandonou >= 50:
        risco = "Moderado"
    elif prob_abandonou >= 30:
        risco = "Baixo"
    else:
        risco = "Muito Baixo"

    # ── Causas textuais ──
    causas = []
    if student_data.get("Repetiu_Ano") == "Sim":
        causas.append("Histórico de repetição de ano.")
    reprovacoes = int(student_data.get("Numero_Reprovacoes", 0))
    if reprovacoes >= 1:
        causas.append(f"{reprovacoes} reprovação(ões) registada(s).")
    if student_data.get("Trabalha") == "Sim":
        causas.append("O estudante exerce actividade laboral.")
    if student_data.get("Bullying") == "Sim":
        causas.append("Existem registos de bullying.")
    distancia = float(student_data.get("Distancia_Escola_km", 0))
    if distancia > 5:
        causas.append(f"Distância elevada até à escola ({distancia} km).")
    if student_data.get("Apoio_Familiar") in ["Não", "Baixo"]:
        causas.append("Baixo apoio familiar identificado.")
    if student_data.get("Acesso_Internet") == "Não":
        causas.append("Sem acesso à internet.")
    if student_data.get("Renda_Familiar") == "Baixa":
        causas.append("Renda familiar baixa.")
    if not causas:
        causas.append("Nenhum factor de risco relevante identificado.")

    # ── SHAP local: calcula e devolve dicts para o agente ──
    import shap
    from src.config import RANDOM_FOREST_PATH, SCALER_PATH

    X_array = np.array(X.toarray() if hasattr(X, "toarray") else X)
    feature_names = preprocessor.get_feature_names_out()

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_array)

    if isinstance(shap_values, list):
        vals = shap_values[1][0]
    elif shap_values.ndim == 3:
        vals = shap_values[0, :, 1]
    else:
        vals = shap_values[0]

    # Agrupa por variável original (soma colunas OneHot da mesma variável)
    agrupado = {}
    for i, fname in enumerate(feature_names):
        pertence = any(chave.lower() in fname.lower() for chave in VARIAVEIS_FORMULARIO)
        if pertence:
            nome = limpar_nome(fname)
            agrupado[nome] = agrupado.get(nome, 0) + float(vals[i])

    shap_risco     = {k: round(v, 4) for k, v in agrupado.items() if v >  0.001}
    shap_protector = {k: round(abs(v), 4) for k, v in agrupado.items() if v < -0.001}

    # Gera gráfico pizza local
    generate_local_explanation(X_array, agrupado)

    return {
        "classe":          classe,
        "probabilidade":   prob_abandonou,
        "prob_estudando":  prob_estudando,
        "risco":           risco,
        "causas":          causas,
        "shap_risco":      shap_risco,
        "shap_protector":  shap_protector,
    }
