from psi_analyzer import PSIAnalyzer
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from matplotlib import pyplot as plt
import seaborn as sns


def simulate_model_data(model_name: str, start_year: int = 2023) -> pd.DataFrame:
    """Simula scores e variável resposta para diferentes ambientes."""
    rng = np.random.default_rng(seed=42 if model_name == "modelo_a" else 99)

    def generate_block(env, year, month, loc, size=1000):
        scores = np.clip(rng.normal(loc=loc, scale=0.1, size=size), 0.01, 0.99)
        vr = rng.binomial(1, p=scores)
        return pd.DataFrame(
            {
                "model": model_name,
                "env": env,
                "year": year,
                "month": month,
                "score": scores,
                "vr": vr,
            }
        )

    data = []
    for month in range(1, 7):
        data.append(generate_block("DEV", start_year, month, 0.5))

    for month in range(7, 13):
        loc = 0.5 if model_name == "modelo_a" else 0.55
        data.append(generate_block("OOT", start_year, month, loc))

    for month in range(1, 13):
        year = start_year + 1
        loc = 0.5 if model_name == "modelo_a" else 0.5 + 0.02 * month
        data.append(generate_block("PRD", year, month, loc))

    return pd.concat(data, ignore_index=True)



if __name__ == "__main__":
    
    

    spark = SparkSession.builder.getOrCreate()
    # Dados com binagem simples: 2 faixas (score baixo e alto)
        # -----------------------
    # Teste 1
    # -----------------------
    data = pd.DataFrame({
        'model': ['modelo_teste'] * 20,
        'env': ['DEV'] * 10 + ['OOT'] * 10,
        'score': [0.1]*5 + [0.9]*5 + [0.1]*2 + [0.9]*8,
        'year_month': ['2023-01'] * 20
    })

    # Converter para Spark DataFrame
    spdf = spark.createDataFrame(data)
    spdf = spdf.withColumn("year", F.year(F.to_date(spdf.year_month, "yyyy-MM")))
    spdf = spdf.withColumn("month", F.month(F.to_date(spdf.year_month, "yyyy-MM")))

    # -----------------------
    # Calcular PSI manualmente
    # -----------------------

    # Binarização:
    # DEV: 5 scores em [0.1] → faixa 1, 5 em [0.9] → faixa 2 → proporção = [0.5, 0.5]
    # OOT: 2 em [0.1] → faixa 1, 8 em [0.9] → faixa 2 → proporção = [0.2, 0.8]

    p = np.array([0.5, 0.5])
    q = np.array([0.2, 0.8])
    psi_manual = np.sum((p - q) * np.log(p / q))  # Deve dar aproximadamente 0.0915

    # -----------------------
    # Usar PSIAnalyzer
    # -----------------------

    analyzer = PSIAnalyzer(
        df=spdf,
        model_col='model',
        env_col='env',
        score_col='score',
        year_col='year',
        month_col='month',
        n_bins=2
    )

    psi_df = analyzer.compute_psi()
    psi_df.show()
    psi_result = psi_df.select("psi").collect()[0]["psi"]

    # -----------------------
    # Verificação
    # -----------------------

    assert abs(psi_result - psi_manual) < 1e-4, f"PSI diferente: {psi_result} vs {psi_manual}"
    print(f"Teste passou! PSI calculado = {psi_result:.5f}")

#     # ======================
#     # Simulação dos dados
#     # ======================
#     # Dados simulados
#     df = pd.concat(
#         [simulate_model_data("modelo_a"), simulate_model_data("modelo_b")],
#         ignore_index=True,
#     )

#     # ======================
#     # Spark Session
#     # ======================
#     spark = SparkSession.builder.appName("PSIAnalysis").getOrCreate()
#     spkdf = spark.createDataFrame(df)

#     # ======================
#     # Cálculo do PSI
#     # ======================
#     psi_analysis = PSIAnalyzer(
#         df = spkdf,
#         model_col="model",
#         env_col="env",
#         score_col="score",
#         year_col="year",
#         month_col="month"
#     )
#     df_psi = psi_analysis.compute_psi()
    
    
#     fig, ax = plt.subplots(figsize=(7, 4))

#     # Plot principal
#     sns.lineplot(
#         x="data",
#         y="psi",
#         hue="model",
#         style="env",
#         data=df_psi.toPandas(),
#         dashes=False,
#         markers=True,
#         markeredgecolor="w",
#     )

#     plt.axhline(0.1, color="orange", linestyle="--", label="limite moderado")
#     plt.axhline(0.25, color="red", linestyle="--", label="limite crítico")
#     plt.title("Evolução do PSI")
#     plt.xlabel("Data")
#     plt.ylabel("PSI")


# plt.xticks(rotation=90)


# # remover upper and right axis and set on the grid
# ax.spines["right"].set_visible(False)
# ax.spines["top"].set_visible(False)
# ax.yaxis.grid(True, linestyle="-", linewidth=0.2)
# ax.xaxis.grid(True, linestyle="-", linewidth=0.2)
# plt.tight_layout()
# plt.legend()
# plt.show()
