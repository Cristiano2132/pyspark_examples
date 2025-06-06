# psi_analyzer.py
from typing import Optional
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.functions import udf, col, count, lit, to_date, concat_ws, lpad


def assign_decile(score: float, cuts: Optional[list]) -> Optional[int]:
    if cuts is None:
        return None
    for i, c in enumerate(cuts):
        if score <= c:
            return i + 1
    return len(cuts) + 1


def label_decile(score: float, cuts: Optional[list]) -> Optional[str]:
    if cuts is None:
        return None
    for i, c in enumerate(cuts):
        if score <= c:
            return f"<{c:.4f}" if i == 0 else f"{cuts[i-1]:.4f} - {c:.4f}"
    return f">{cuts[-1]:.4f}"


assign_decile_udf = udf(assign_decile, IntegerType())
label_decile_udf = udf(label_decile, StringType())

# This class `PSIAnalyzer2` in Python is designed to compute the Population Stability Index (PSI) for
# a given DataFrame based on specified columns and parameters.

class PSIAnalyzer:
    def __init__(
        self,
        df: DataFrame,
        model_col: str = "model",
        env_col: str = "env",
        score_col: str = "score",
        year_col: str = "year",
        month_col: str = "month",
        reference_env: str = "DEV",
        n_bins: int = 10,
    ) -> None:
        self.df = df
        self.model_col = model_col
        self.env_col = env_col
        self.score_col = score_col
        self.year_col = year_col
        self.month_col = month_col
        self.n_bins = n_bins
        self.reference_env = reference_env

    def _get_score_cuts(self) -> DataFrame:
        percentiles = [str(i / self.n_bins) for i in range(1, self.n_bins)]
        return (
            self.df.filter(col(self.env_col) == self.reference_env)
            .groupBy(self.model_col)
            .agg(
                F.expr(
                    f"percentile_approx({self.score_col}, array({','.join(percentiles)}), 10000)"
                ).alias("cuts")
            )
        )

    def _assign_deciles_and_labels(self, df: DataFrame, cuts_df: DataFrame) -> DataFrame:
        return df.join(cuts_df, on=self.model_col, how="left").withColumn(
            "faixa_score", assign_decile_udf(col(self.score_col), col("cuts"))
        ).withColumn(
            "label_faixa_score", label_decile_udf(col(self.score_col), col("cuts"))
        )

    def _calculate_proportions(self, df: DataFrame, is_reference: bool) -> DataFrame:
        if is_reference:
            counts = (
                df.filter(col(self.env_col) == self.reference_env)
                .groupBy(self.model_col, self.env_col, "faixa_score")
                .agg(count("*").alias("count"))
            )
            window = Window.partitionBy(self.model_col, self.env_col)
            return counts.withColumn("total", F.sum("count").over(window)).withColumn(
                "proporcao_ref", col("count") / col("total")
            )
        else:
            counts = (
                df.filter(col(self.env_col) != self.reference_env)
                .groupBy(
                    self.model_col,
                    self.env_col,
                    "faixa_score",
                    self.year_col,
                    self.month_col,
                )
                .agg(count("*").alias("count"))
            )
            window = Window.partitionBy(self.model_col, self.env_col, self.year_col, self.month_col)
            return counts.withColumn("total", F.sum("count").over(window)).withColumn(
                "proporcao", col("count") / col("total")
            )

    def _calculate_psi(self, df_out: DataFrame, df_ref: DataFrame) -> DataFrame:
        df_joined = df_out.join(
            df_ref.select(self.model_col, "faixa_score", "proporcao_ref"),
            on=[self.model_col, "faixa_score"],
            how="left",
        )

        safe_ref = F.when(col("proporcao_ref") == 0, lit(1e-6)).otherwise(col("proporcao_ref"))
        safe_out = F.when(col("proporcao") == 0, lit(1e-6)).otherwise(col("proporcao"))

        df_joined = df_joined.withColumn(
            "psi_component", (safe_ref - safe_out) * F.log(safe_ref / safe_out)
        )

        return df_joined.groupBy(
            self.model_col, self.env_col, self.year_col, self.month_col
        ).agg(
            F.sum("psi_component").alias("psi")
        ).withColumn(
            "data",
            to_date(
                concat_ws(
                    "-",
                    col(self.year_col),
                    lpad(col(self.month_col), 2, "0"),
                    lit("01"),
                ),
                "yyyy-MM-dd",
            ),
        )

    def compute_psi(self) -> DataFrame:
        df = self.df.filter(col(self.score_col).isNotNull())
        cuts_df = self._get_score_cuts()
        df_with_deciles = self._assign_deciles_and_labels(df, cuts_df)
        df_ref = self._calculate_proportions(df_with_deciles, is_reference=True)
        df_out = self._calculate_proportions(df_with_deciles, is_reference=False)
        return self._calculate_psi(df_out, df_ref)