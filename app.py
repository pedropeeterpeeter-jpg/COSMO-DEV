import os
from flask import Flask, render_template, request, session, jsonify
import requests
from dotenv import load_dotenv

from src.predict import predict_student
from src.dashboard_stats import get_dashboard_stats
from src.agent import chat_com_agente, mensagem_inicial

import webbrowser

load_dotenv()

print("CHAVE:", os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "xai-angola-2025")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    stats = get_dashboard_stats()
    return render_template("dashboard.html", stats=stats)


@app.route("/predict", methods=["POST"])
def predict():
    student = {
        "Idade":               int(request.form["idade"]),
        "Sexo":                request.form["sexo"],
        "Provincia":           request.form["provincia"],
        "Municipio":           request.form["municipio"],
        "Zona":                request.form["zona"],
        "Tipo_Instituicao":    request.form["tipo_instituicao"],
        "Nivel_Ensino":        request.form["nivel_ensino"],
        "Classe":              request.form["classe"],
        "Renda_Familiar":      request.form["renda_familiar"],
        "Numero_Irmaos":       int(request.form["numero_irmaos"]),
        "Distancia_Escola_km": float(request.form["distancia"]),
        "Meio_Transporte":     request.form["transporte"],
        "Trabalha":            request.form["trabalha"],
        "Repetiu_Ano":         request.form["repetiu"],
        "Numero_Reprovacoes":  int(request.form["reprovacoes"]),
        "Apoio_Familiar":      request.form["apoio"],
        "Bullying":            request.form["bullying"],
        "Acesso_Internet":     request.form["internet"],
        "Pretende_Retornar":   request.form["retornar"]
    }

    resultado = predict_student(student)

    # Guarda na sessão para todas as páginas seguintes
    session["student_data"]    = student
    session["resultado"]       = resultado
    session["shap_risco"]      = resultado.get("shap_risco", {})
    session["shap_protector"]  = resultado.get("shap_protector", {})

    # ── NOVO FLUXO: após previsão vai directo ao chat ──
    return render_template("chat.html", resultado=resultado)


@app.route("/shap_global")
def shap_global():
    """Página SHAP Global — acessível a partir do chat."""
    resultado = session.get("resultado")
    if not resultado:
        return render_template("index.html")
    return render_template("shap_global.html", resultado=resultado)


@app.route("/shap_local")
def shap_local():
    """Página SHAP Local (XAI completo) — acessível a partir do SHAP global."""
    resultado = session.get("resultado")
    if not resultado:
        return render_template("index.html")
    return render_template("shap_local.html", resultado=resultado)


@app.route("/chat")
def chat():
    resultado = session.get("resultado")
    if not resultado:
        return render_template("index.html")
    return render_template("chat.html", resultado=resultado)


@app.route("/chat_inicio", methods=["POST"])
def chat_inicio():
    student_data   = session.get("student_data", {})
    resultado      = session.get("resultado", {})
    shap_risco     = session.get("shap_risco", {})
    shap_protector = session.get("shap_protector", {})
    resposta = mensagem_inicial(student_data, resultado, shap_risco, shap_protector)
    return jsonify({"resposta": resposta})


@app.route("/chat_mensagem", methods=["POST"])
def chat_mensagem():
    data           = request.get_json()
    mensagem       = data.get("mensagem", "")
    historico      = data.get("historico", [])
    student_data   = session.get("student_data", {})
    resultado      = session.get("resultado", {})
    shap_risco     = session.get("shap_risco", {})
    shap_protector = session.get("shap_protector", {})
    resposta = chat_com_agente(
        historico, student_data, resultado,
        shap_risco, shap_protector, mensagem
    )
    return jsonify({"resposta": resposta})


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)
