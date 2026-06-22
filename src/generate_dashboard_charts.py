import matplotlib.pyplot as plt

from src.data_loader import load_data

from src.config import IMAGE_DIR


def generate_charts():

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    # Situação Escolar

    plt.figure(figsize=(6,4))

    df["Situacao_Escolar"].value_counts().plot(
        kind="bar"
    )

    plt.title(
        "Situação Escolar"
    )

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "situacao_escolar.png"
    )

    plt.close()

    # Sexo

    plt.figure(figsize=(6,4))

    df["Sexo"].value_counts().plot(
        kind="bar"
    )

    plt.title(
        "Distribuição por Sexo"
    )

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "sexo.png"
    )

    plt.close()

    # Província

    plt.figure(figsize=(10,5))

    df["Provincia"].value_counts().plot(
        kind="bar"
    )

    plt.title(
        "Distribuição por Província"
    )

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "provincia.png"
    )

    plt.close()

    print("Gráficos gerados.")


if __name__ == "__main__":

    generate_charts()
    