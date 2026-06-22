import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import IMAGE_DIR


def generate_local_explanation(student_array, agrupado: dict):
    """
    Gera gráfico SHAP Local em formato PIZZA DUPLA.

    Parâmetros:
        student_array : numpy array (1, n_features) — já pré-processado
        agrupado      : dict {nome_variavel: valor_shap} calculado em predict.py
                        Valores positivos → aumentam risco de abandono
                        Valores negativos → reduzem risco de abandono

    O agrupado já vem filtrado para apenas as variáveis do formulário,
    com nomes limpos (ex: "Apoio Familiar", não "cat__Apoio_Familiar_Baixo").

    Guarda em static/imagens/shap_local.png
    """

    Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)

    # Separa factores de risco (SHAP > 0) e protectores (SHAP < 0)
    risco      = {k: v       for k, v in agrupado.items() if v >  0.001}
    protector  = {k: abs(v)  for k, v in agrupado.items() if v < -0.001}

    cores_risco = [
        "#E74C3C", "#C0392B", "#E67E22", "#D35400",
        "#F39C12", "#A93226", "#CB4335", "#B03A2E"
    ]
    cores_protector = [
        "#27AE60", "#1E8449", "#2ECC71", "#17A589",
        "#148F77", "#1ABC9C", "#0E6655", "#0B5345"
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Factores que influenciaram a previsão deste estudante",
        fontsize=13, fontweight="bold", y=1.01
    )

    # ── Pizza esquerda: RISCO ──────────────────────────────────────
    if risco:
        labels_r = list(risco.keys())
        vals_r   = list(risco.values())
        wedges_r, _, autotexts_r = axes[0].pie(
            vals_r,
            labels=None,
            autopct="%1.1f%%",
            startangle=140,
            colors=cores_risco[:len(labels_r)],
            pctdistance=0.75,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5}
        )
        for at in autotexts_r:
            at.set_fontsize(8.5)
            at.set_color("white")
            at.set_fontweight("bold")
        axes[0].legend(
            wedges_r, labels_r,
            title="Factores",
            title_fontsize=9,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            fontsize=8.5,
            ncol=2,
            frameon=True
        )
    else:
        axes[0].text(
            0.5, 0.5, "Sem factores\nde risco identificados",
            ha="center", va="center", fontsize=11,
            color="#27AE60", transform=axes[0].transAxes
        )
        axes[0].axis("off")

    axes[0].set_title(
        "Aumentam o risco de abandono",
        fontsize=11, color="#C0392B", fontweight="bold", pad=12
    )

    # ── Pizza direita: PROTECTORES ─────────────────────────────────
    if protector:
        labels_p = list(protector.keys())
        vals_p   = list(protector.values())
        wedges_p, _, autotexts_p = axes[1].pie(
            vals_p,
            labels=None,
            autopct="%1.1f%%",
            startangle=140,
            colors=cores_protector[:len(labels_p)],
            pctdistance=0.75,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5}
        )
        for at in autotexts_p:
            at.set_fontsize(8.5)
            at.set_color("white")
            at.set_fontweight("bold")
        axes[1].legend(
            wedges_p, labels_p,
            title="Factores",
            title_fontsize=9,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            fontsize=8.5,
            ncol=2,
            frameon=True
        )
    else:
        axes[1].text(
            0.5, 0.5, "Sem factores\nprotectores identificados",
            ha="center", va="center", fontsize=11,
            color="#C0392B", transform=axes[1].transAxes
        )
        axes[1].axis("off")

    axes[1].set_title(
        "Reduzem o risco de abandono",
        fontsize=11, color="#1E8449", fontweight="bold", pad=12
    )

    plt.tight_layout()
    plt.savefig(
        Path(IMAGE_DIR) / "shap_local.png",
        bbox_inches="tight",
        dpi=130
    )
    plt.close()
    print("SHAP Local (pizza) guardado.")
