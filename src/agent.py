import os
import requests

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


def formatar_dados_estudante(student_data: dict) -> str:
    linhas = []
    for chave, valor in student_data.items():
        nome = NOMES_LEGIVEIS.get(chave, chave)
        linhas.append(f"  - {nome}: {valor}")
    return "\n".join(linhas)


def formatar_shap(shap_risco: dict, shap_protector: dict) -> str:
    texto = ""
    total = sum(shap_risco.values()) + sum(shap_protector.values()) + 0.0001
    if shap_risco:
        texto += "Factores que AUMENTAM o risco de abandono:\n"
        for var, peso in sorted(shap_risco.items(), key=lambda x: -x[1]):
            pct = round(peso / total * 100, 1)
            texto += f"  - {var}: {pct}% de influência\n"
    if shap_protector:
        texto += "Factores que REDUZEM o risco de abandono:\n"
        for var, peso in sorted(shap_protector.items(), key=lambda x: -x[1]):
            pct = round(peso / total * 100, 1)
            texto += f"  - {var}: {pct}% de protecção\n"
    return texto if texto else "Nenhum factor relevante identificado."


def construir_sistema_prompt(student_data, resultado, shap_risco, shap_protector) -> str:
    dados_fmt = formatar_dados_estudante(student_data)
    shap_fmt  = formatar_shap(shap_risco, shap_protector)
    classe    = resultado.get("classe", "Desconhecido")
    prob      = resultado.get("probabilidade", 0)
    risco     = resultado.get("risco", "Desconhecido")

    return f"""És um especialista em educação e inteligência artificial explicável (XAI), \
integrado num sistema de previsão de abandono escolar .

O modelo analisou um estudante com os seguintes dados:

DADOS DO ESTUDANTE:
{dados_fmt}

RESULTADO DO MODELO:
  - Classe prevista: {classe}
  - Probabilidade de abandono: {prob}%
  - Nível de risco: {risco}

EXPLICAÇÃO DO MODELO (o que influenciou a decisão):
{shap_fmt}

A tua missão:
1. Explicar em português simples o que o modelo decidiu e porquê.
2. Identificar os factores mais críticos para este estudante.
3. Sugerir acções concretas para a escola ou família.
4. Responder perguntas de acompanhamento com base nos dados reais.

Regras:
- Linguagem simples, sem termos técnicos como SHAP ou Random Forest.
- Sê empático, construtivo, nunca alarmista.
- Máximo 4 parágrafos por resposta.
- Baseia-te sempre nos dados reais deste estudante.
- Escreve sempre em português de Angola.
"""


def chamar_api(sistema: str, mensagens: list) -> str:
    """
    Chama a API do Google Gemini via requests.
    Modelo: gemini-2.5-flash — gratuito e sem cartão.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "Erro: chave GEMINI_API_KEY não encontrada no ficheiro .env"

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )

    # Gemini usa formato próprio: role "user"/"model" (não "assistant")
    # Concatena o prompt de sistema na primeira mensagem do utilizador
    conteudos = []
    primeiro = True
    for msg in mensagens:
        role = "user" if msg["role"] == "user" else "model"
        texto = msg["content"]
        if primeiro and msg["role"] == "user":
            texto = sistema + "\n\n---\n\n" + texto
            primeiro = False
        conteudos.append({
            "role": role,
            "parts": [{"text": texto}]
        })

    # Se não houver mensagens do utilizador ainda, cria uma
    if not conteudos:
        conteudos = [{
            "role": "user",
            "parts": [{"text": sistema}]
        }]

    payload = {
        "contents": conteudos,
        "generationConfig": {
            "maxOutputTokens": 1024,
            "temperature":     0.7
        }
    }

    try:
        resp = requests.post(
            url,
            json=payload,
            timeout=30
        )

        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

        elif resp.status_code == 400:
            return f"Erro no pedido: verifica a chave GEMINI_API_KEY no ficheiro .env"

        elif resp.status_code == 403:
            return "Acesso negado: a chave API não tem permissão. Gera uma nova em aistudio.google.com/apikey"

        elif resp.status_code == 429:
            return "Limite temporário atingido. Aguarda 1 minuto e tenta novamente."

        else:
            return f"Erro da API ({resp.status_code}): {resp.text[:300]}"

    except requests.exceptions.ConnectionError:
        return "Erro de ligação: verifica a tua ligação à internet."
    except requests.exceptions.Timeout:
        return "Tempo de resposta esgotado. Tenta novamente."
    except KeyError:
        return f"Resposta inesperada da API: {resp.text[:300]}"
    except Exception as e:
        return f"Erro inesperado: {str(e)}"


def chat_com_agente(
    historico, student_data, resultado,
    shap_risco, shap_protector, mensagem_utilizador
) -> str:
    sistema   = construir_sistema_prompt(student_data, resultado, shap_risco, shap_protector)
    mensagens = historico + [{"role": "user", "content": mensagem_utilizador}]
    return chamar_api(sistema, mensagens)


def mensagem_inicial(student_data, resultado, shap_risco, shap_protector) -> str:
    sistema   = construir_sistema_prompt(student_data, resultado, shap_risco, shap_protector)
    mensagens = [{
        "role": "user",
        "content": (
            "Analisa os dados deste estudante e explica o resultado da previsão. "
            "Começa com uma conclusão clara sobre o risco, depois explica os factores "
            "principais e termina com recomendações concretas para a escola ou família."
        )
    }]
    return chamar_api(sistema, mensagens)
