from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StringType


# UDF para determinar a faixa de score com base na lista de cuts
def definir_faixa(score, cuts):
    """
    Determine the score band based on the provided cuts.

    Args:
        score (float): The score value to evaluate
        cuts (list): List of cut points defining the score bands

    Returns:
        int: The band number (1 to len(cuts) + 1)
    """
    if cuts is None:
        return None
    for i, limite in enumerate(cuts):
        if score <= limite:
            return i + 1
    return len(cuts) + 1  # Última faixa


aplicar_faixa_udf = udf(definir_faixa, IntegerType())


def gerar_label(score, cuts):
    """
    Generate a label string representing the score band range.

    Args:
        score (float): The score value to evaluate
        cuts (list): List of cut points defining the score bands

    Returns:
        str: A string representation of the score band range (e.g. "0.000 - 0.250")
    """
    if cuts is None or len(cuts) == 0:
        return None
    limites = [0] + sorted(cuts) + [1]
    for i in range(len(limites) - 1):
        if limites[i] <= score <= limites[i + 1]:
            return f"{limites[i]:.3f} - {limites[i + 1]:.3f}"
    return None


aplicar_label_udf = udf(gerar_label, StringType())


class FaixaScoreCalculator:
    """
    A class to calculate statistics for different score bands.

    This class processes a DataFrame containing scores and calculates various
    statistics for predefined score bands, such as risk, event rates, and
    observation rates.
    """

    def __init__(
        self,
        df: DataFrame,
        ano_mes_column: str,
        vr_column: str,
        score_column: str,
        modelo_column: str,
        ambiente_column: str,
        dev_code: any,
        cuts: list = None,
    ):
        """
        Initialize the FaixaScoreCalculator.

        Args:
            df (DataFrame): The input DataFrame containing the data
            ano_mes_column (str): Column name for year-month
            vr_column (str): Column name for the event indicator
            score_column (str): Column name for the score
            modelo_column (str): Column name for the model identifier
            ambiente_column (str): Column name for the environment
            dev_code (any): Code identifying the development environment
            cuts (list, optional): List of cut points for score bands. Defaults to None.
        """
        self.cuts = cuts
        self.df = df
        self.ano_mes_column = ano_mes_column
        self.vr_column = vr_column
        self.score_column = score_column
        self.modelo_column = modelo_column
        self.ambiente_column = ambiente_column
        self.dev_code = dev_code

    def set_cuts(self, n_faixas_score: int) -> list:
        """
        Set the cut points for score bands based on the number of bands.

        Args:
            n_faixas_score (int): Number of score bands to create

        Returns:
            list: The list of cut points
        """
        self.cuts = [i / n_faixas_score for i in range(1, n_faixas_score)]
        return self.cuts

    def _get_faixas_score_dev(self):
        """
        Calculate score bands based on the development environment data.

        This method calculates the percentiles of scores in the development
        environment and applies them to create score bands for all data.

        Raises:
            ValueError: If cuts are not defined
        """
        if self.cuts is None:
            raise ValueError(
                "Faixas de score não definidas. Utilize o método set_cuts para definir as faixas."
            )

        df_dev_cuts = (
            self.df.filter(F.col(self.ambiente_column) == self.dev_code)
            .groupBy(self.modelo_column)
            .agg(
                F.percentile_approx(self.score_column, self.cuts, 10000).alias(
                    "cuts_score"
                )
            )
        )

        self.df = self.df.join(df_dev_cuts, on=self.modelo_column, how="left")

        # Aplicar a UDF para criar a coluna faixa_score
        self.df = self.df.withColumn(
            "faixa_score",
            aplicar_faixa_udf(F.col(self.score_column), F.col("cuts_score")),
        )

        # Aplicar a UDF para criar a coluna de label
        self.df = self.df.withColumn(
            "faixa_score_label",
            aplicar_label_udf(F.col(self.score_column), F.col("cuts_score")),
        )

    def get_faixas_score(self):
        """
        Calculate statistics for each score band.

        This method groups the data by score bands and calculates various
        statistics including count, event count, risk, event rates, and
        observation rates for each band.

        Returns:
            DataFrame: A DataFrame containing statistics for each score band
        """
        # Validate required columns exist
        required_columns = [
            self.modelo_column,
            self.ambiente_column,
            self.ano_mes_column,
            self.vr_column,
            self.score_column,
        ]
        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        self._get_faixas_score_dev()
        janela_periodo = Window.partitionBy(
            self.modelo_column, self.ambiente_column, self.ano_mes_column
        )

        spkdf_faixas_score = (
            self.df.groupBy(
                self.modelo_column,
                self.ambiente_column,
                self.ano_mes_column,
                "faixa_score",
                "faixa_score_label",
            )
            .agg(
                F.count("*").alias("count_por_faixa"),
                F.sum(self.vr_column).alias("eventos_por_faixa"),
            )
            .withColumn(
                "count_por_periodo", F.sum("count_por_faixa").over(janela_periodo)
            )
            .withColumn(
                "risco_faixa",
                F.when(
                    F.col("count_por_faixa") > 0,
                    F.col("eventos_por_faixa") / F.col("count_por_faixa"),
                ).otherwise(0),
            )
            .withColumn(
                "taxa_eventos_por_faixa",
                F.when(
                    F.col("count_por_periodo") > 0,
                        F.col("eventos_por_faixa") / F.col("count_por_periodo"),
                ).otherwise(0),
            )
            .withColumn(
                "taxa_obs_por_faixa",
                F.when(
                    F.col("count_por_periodo") > 0,
                        F.col("count_por_faixa") / F.col("count_por_periodo"),
                ).otherwise(0),
            )
            .orderBy(self.modelo_column, self.ano_mes_column, "faixa_score")
        )

        return spkdf_faixas_score
    