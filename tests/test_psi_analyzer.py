import unittest
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from psi_analyzer import PSIAnalyzer  # ajuste conforme sua estrutura


class TestPSIAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[1]").appName("PSIUnitTest").getOrCreate()

    def test_psi_computation_basic_case(self):
        # --- Arrange ---
        data = pd.DataFrame({
            'model': ['modelo_teste'] * 20,
            'env': ['DEV'] * 10 + ['OOT'] * 10,
            'score': [0.1]*5 + [0.9]*5 + [0.1]*2 + [0.9]*8,
            'year_month': ['2023-01'] * 20
        })

        spdf = self.spark.createDataFrame(data)
        spdf = spdf.withColumn("year", F.year(F.to_date(spdf.year_month, "yyyy-MM")))
        spdf = spdf.withColumn("month", F.month(F.to_date(spdf.year_month, "yyyy-MM")))

        p = np.array([0.5, 0.5])
        q = np.array([0.2, 0.8])
        psi_manual = np.sum((p - q) * np.log(p / q))  # ~0.0915

        analyzer = PSIAnalyzer(
            df=spdf,
            model_col='model',
            env_col='env',
            score_col='score',
            year_col='year',
            month_col='month',
            n_bins=2
        )

        # --- Act ---
        psi_df = analyzer.compute_psi()
        psi_result = psi_df.select("psi").collect()[0]["psi"]

        # --- Assert ---
        self.assertAlmostEqual(psi_result, psi_manual, places=4, msg=f"PSI should be {psi_manual:.5f}, got {psi_result:.5f}")

    def test_empty_dataframe(self):
        # --- Arrange ---
        empty_df = self.spark.createDataFrame([], schema="model string, env string, score double, year int, month int")

        analyzer = PSIAnalyzer(
            df=empty_df,
            model_col='model',
            env_col='env',
            score_col='score',
            year_col='year',
            month_col='month',
            n_bins=2
        )

        # --- Act ---
        result_df = analyzer.compute_psi()

        # --- Assert ---
        self.assertEqual(result_df.count(), 0, "Resultado deve ser vazio para entrada vazia.")

    def test_constant_score_should_yield_zero_psi(self):
        # --- Arrange ---
        data = pd.DataFrame({
            'model': ['modelo_teste'] * 20,
            'env': ['DEV'] * 10 + ['OOT'] * 10,
            'score': [0.5] * 20,
            'year_month': ['2023-01'] * 20
        })
        spdf = self.spark.createDataFrame(data)
        spdf = spdf.withColumn("year", F.year(F.to_date(spdf.year_month, "yyyy-MM")))
        spdf = spdf.withColumn("month", F.month(F.to_date(spdf.year_month, "yyyy-MM")))

        analyzer = PSIAnalyzer(
            df=spdf,
            model_col='model',
            env_col='env',
            score_col='score',
            year_col='year',
            month_col='month',
            n_bins=2
        )

        # --- Act ---
        psi_df = analyzer.compute_psi()
        psi_value = psi_df.select("psi").collect()[0]["psi"]

        # --- Assert ---
        self.assertAlmostEqual(psi_value, 0.0, places=5, msg="PSI deve ser zero para distribuição idêntica")

    def test_shifted_distribution_should_yield_low_psi(self):
        # --- Arrange ---
        data = pd.DataFrame({
            'model': ['modelo_teste'] * 600,
            'env': ['DEV'] * 200 + ['OOT'] * 400,
            'score': list(np.random.normal(0.4, 0.2, 200)) + list(np.random.normal(0.4, 0.2, 400)),
            'year_month': ['2023-01'] * 200 + ['2023-02'] * 400 
        })
        spdf = self.spark.createDataFrame(data)
        spdf = spdf.withColumn("year", F.year(F.to_date(spdf.year_month, "yyyy-MM")))
        spdf = spdf.withColumn("month", F.month(F.to_date(spdf.year_month, "yyyy-MM")))

        analyzer = PSIAnalyzer(
            df=spdf,
            model_col='model',
            env_col='env',
            score_col='score',
            year_col='year',
            month_col='month',
            n_bins=10
        )

        # --- Act ---
        psi_df = analyzer.compute_psi()
        psi_value = psi_df.select("psi").collect()[0]["psi"]

        # --- Assert ---
        self.assertIsNotNone(psi_value)
        self.assertTrue(psi_value< 0.3)

    def test_shifted_distribution_should_yield_high_psi(self):
        # --- Arrange ---
        data = pd.DataFrame({
            'model': ['modelo_teste'] * 20,
            'env': ['DEV'] * 10 + ['OOT'] * 10,
            'score': list(np.random.normal(0.4, 0.1, 10)) + list(np.random.normal(0.7, 0.1, 10)),
            'year_month': ['2023-01'] * 20
        })
        spdf = self.spark.createDataFrame(data)
        spdf = spdf.withColumn("year", F.year(F.to_date(spdf.year_month, "yyyy-MM")))
        spdf = spdf.withColumn("month", F.month(F.to_date(spdf.year_month, "yyyy-MM")))

        analyzer = PSIAnalyzer(
            df=spdf,
            model_col='model',
            env_col='env',
            score_col='score',
            year_col='year',
            month_col='month',
            n_bins=2
        )

        # --- Act ---
        psi_df = analyzer.compute_psi()
        psi_value = psi_df.select("psi").collect()[0]["psi"]

        # --- Assert ---
        self.assertIsNotNone(psi_value)
        self.assertGreater(psi_value, 0.2)


    def test_shifted_distribution_should_yield_low_and_high_psi(self):
        # --- Arrange ---
        list_ = list(np.random.normal(0.4, 0.1, 100))
        data = pd.DataFrame({
                    'model': ['a'] * 300,
                    'env': ['DEV'] * 100 + ['PRD'] * 200,
                    'score': list_ + list_ + list(
                        np.random.normal(0.7, 0.1, 100)
                        ),
                    'year_month': ['2023-01'] * 100 + ['2023-02'] * 100 + ['2023-03'] * 100 
            })
        spdf = self.spark.createDataFrame(data)
        spdf = spdf.withColumn("year", F.year(F.to_date(spdf.year_month, "yyyy-MM")))
        spdf = spdf.withColumn("month", F.month(F.to_date(spdf.year_month, "yyyy-MM")))

        analyzer = PSIAnalyzer(
            df=spdf,
            model_col='model',
            env_col='env',
            score_col='score',
            year_col='year',
            month_col='month',
            n_bins=2
        )
        # --- Act ---
        psi_df = analyzer.compute_psi()
        psi_value_1 = psi_df.select("psi").collect()[0]["psi"]
        psi_value_2 = psi_df.select("psi").collect()[1]["psi"]

        # --- Assert ---
        self.assertIsNotNone(psi_value_1)
        self.assertIsNotNone(psi_value_2)
        self.assertTrue(psi_value_1< psi_value_2)
        self.assertTrue(psi_value_1==0)
        self.assertTrue(psi_value_2>0.0)
        self.assertTrue(psi_value_2>0.3)

    def test_null_scores_are_ignored(self):
        # --- Arrange ---
        data = pd.DataFrame({
            'model': ['modelo_teste'] * 22,
            'env': ['DEV'] * 11 + ['OOT'] * 11,
            'score': list(np.random.normal(0.4, 0.1, 10)) + list(np.random.normal(0.7, 0.1, 10)) + [None, None],
            'year_month': ['2023-01'] * 22
        })
        spdf = self.spark.createDataFrame(data)
        spdf = spdf.withColumn("year", F.year(F.to_date(spdf.year_month, "yyyy-MM")))
        spdf = spdf.withColumn("month", F.month(F.to_date(spdf.year_month, "yyyy-MM")))

        analyzer = PSIAnalyzer(
            df=spdf,
            model_col='model',
            env_col='env',
            score_col='score',
            year_col='year',
            month_col='month',
            n_bins=2
        )

        # --- Act ---
        psi_df = analyzer.compute_psi()
        psi_value = psi_df.select("psi").collect()[0]["psi"]

        # --- Assert ---
        self.assertIsNotNone(psi_value)
        self.assertTrue(psi_value > 0, "PSI deve ser calculado ignorando valores nulos")

    def test_multiple_models_are_processed_independently(self):
        # --- Arrange ---
        data = pd.DataFrame({
            'model': ['modelo_a'] * 20 + ['modelo_b'] * 20,
            'env': ['DEV'] * 10 + ['OOT'] * 10 + ['DEV'] * 10 + ['OOT'] * 10,
            'score': [0.1]*5 + [0.9]*5 + [0.1]*2 + [0.9]*8 + [0.3]*5 + [0.7]*5 + [0.4]*3 + [0.6]*7,
            'year_month': ['2023-01'] * 40
        })
        spdf = self.spark.createDataFrame(data)
        spdf = spdf.withColumn("year", F.year(F.to_date(spdf.year_month, "yyyy-MM")))
        spdf = spdf.withColumn("month", F.month(F.to_date(spdf.year_month, "yyyy-MM")))

        analyzer = PSIAnalyzer(
            df=spdf,
            model_col='model',
            env_col='env',
            score_col='score',
            year_col='year',
            month_col='month',
            n_bins=2
        )

        # --- Act ---
        psi_df = analyzer.compute_psi()
        models = psi_df.select("model").distinct().rdd.flatMap(lambda x: x).collect()

        # --- Assert ---
        self.assertIn("modelo_a", models)
        self.assertIn("modelo_b", models)
        self.assertEqual(len(models), 2, "Deve haver PSI calculado para cada modelo separadamente")
        
