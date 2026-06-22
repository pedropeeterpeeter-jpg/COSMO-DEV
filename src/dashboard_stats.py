from src.data_loader import load_data


def get_dashboard_stats():
    """
    Calcula as estatísticas do dataset para o dashboard.

    Devolve um dicionário com as chaves exactas que o
    dashboard.html espera: total, abandonou, estudando,
    taxa, causas.
    """

    df = load_data()

    total     = len(df)
    abandonou = len(df[df["Situacao_Escolar"] == "Abandonou"])
    estudando = len(df[df["Situacao_Escolar"] == "Estudando"])

    taxa = round((abandonou / total) * 100, 1) if total > 0 else 0

    # ─────────────────────────────────────────────────────
    # Causas ordenadas por frequência descendente.
    # Mantemos "Nenhuma" aqui porque o dashboard.html já
    # filtra essa linha com {% if causa != "Nenhuma" %}.
    # head(10) limita a tabela às 10 causas mais frequentes.
    # ─────────────────────────────────────────────────────
    causas = (
        df["Principal_Causa_Abandono"]
        .value_counts()
        .head(10)
        .to_dict()
    )

    # ─────────────────────────────────────────────────────
    # CORRECÇÃO: as chaves devolvidas agora correspondem
    # exactamente ao que o dashboard.html usa:
    #   stats.total, stats.abandonou, stats.estudando,
    #   stats.taxa, stats.causas
    # O código anterior usava "total_estudantes", "abandonos"
    # e "taxa_abandono" — o dashboard ficava em branco.
    # ─────────────────────────────────────────────────────
    return {
        "total":     total,
        "abandonou": abandonou,
        "estudando": estudando,
        "taxa":      taxa,
        "causas":    causas
    }
