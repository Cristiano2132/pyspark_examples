import unittest
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from psi_analyzer import PSIAnalyzer  # ajuste conforme sua estrutura
from psi_analyzer import assign_decile, label_decile



def collect_as_dict(df):
    return [row.asDict() for row in df.collect()]

class TestPSIAssignments(unittest.TestCase):
    def test_assign_decile(self):
        cuts = [0.2, 0.4, 0.6, 0.8]
        self.assertEqual(assign_decile(0.1, cuts), 1)
        self.assertEqual(assign_decile(0.3, cuts), 2)
        self.assertEqual(assign_decile(0.7, cuts), 4)
        self.assertEqual(assign_decile(0.9, cuts), 5)
        self.assertIsNone(assign_decile(0.5, None))

    def test_label_decile(self):
        cuts = [0.2, 0.4, 0.6]
        self.assertEqual(label_decile(0.1, cuts), "<0.2000")
        self.assertEqual(label_decile(0.3, cuts), "0.2000 - 0.4000")
        self.assertEqual(label_decile(0.5, cuts), "0.4000 - 0.6000")
        self.assertEqual(label_decile(0.9, cuts), ">0.6000")
        self.assertIsNone(label_decile(0.5, None))



class TestPSIAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[1]").appName("PSIUnitTest").getOrCreate()

    def test_psi_computation_basic_case(self):
        l_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # --- Arrange ---
        data = pd.DataFrame({
            'model': ['modelo_teste'] * 30,
            'env': ['DEV'] * 10 + ['OOT'] * 10 + ['PRD'] * 10,
            'score': l_values + l_values + [0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3],
            'year_month': ['2023-01'] * 10 + ['2023-02'] * 10 + ['2023-03'] * 10
        })

        spdf = self.spark.createDataFrame(data)
        spdf = spdf.withColumn("year", F.year(F.to_date(spdf.year_month, "yyyy-MM")))
        spdf = spdf.withColumn("month", F.month(F.to_date(spdf.year_month, "yyyy-MM")))

        p = np.array([0.2]*5)
        q = np.array([1, 1e-6, 1e-6, 1e-6, 1e-6])
        
        psi_manual = np.sum((p - q) * np.log(p / q))
        print(psi_manual)

        analyzer = PSIAnalyzer(
            df=spdf,
            model_col='model',
            env_col='env',
            score_col='score',
            year_col='year',
            month_col='month',
            n_bins=5
        )

        # --- Act ---
        psi_df = analyzer.compute_psi()
        psi_result = psi_df.select("psi").collect()[0]["psi"]
        psi1 = psi_df.select("psi").collect()[0]["psi"]
        psi2 = psi_df.select("psi").collect()[1]["psi"]

        # --- Assert ---
        self.assertIsNotNone(psi_result)
        self.assertEqual(psi1, 0)
        self.assertTrue(psi2>0)
        # self.assertEqual(psi2, psi_manual,msg=f"PSI should be {psi_manual:.5f}, got {psi2:.5f}")
        
        # self.assertAlmostEqual(psi_result, psi_manual, places=4, msg=f"PSI should be {psi_manual:.5f}, got {psi_result:.5f}")

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



class TestPSIAnalyzerInternalMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[1]").appName("PSITest").getOrCreate()

        cls.data = [
            ("model1", "DEV", 0.09, 2024, 1),
            ("model1", "DEV", 0.1, 2024, 1),
            ("model1", "DEV", 0.3, 2024, 1),
            ("model1", "DEV", 0.35, 2024, 1),
            ("model1", "DEV", 0.5, 2024, 1),
            ("model1", "DEV", 0.55, 2024, 1),
            ("model1", "DEV", 0.7, 2024, 1),
            ("model1", "DEV", 0.75, 2024, 1),
            ("model1", "DEV", 0.9, 2024, 1),
            ("model1", "DEV", 0.95, 2024, 1),
            ("model1", "PRD", 0.15, 2024, 2),
            ("model1", "PRD", 0.35, 2024, 2),
            ("model1", "PRD", 0.55, 2024, 2),
            ("model1", "PRD", 0.75, 2024, 2),
            ("model1", "PRD", 0.95, 2024, 2),
        ]
        cls.df = cls.spark.createDataFrame(cls.data, ["model", "env", "score", "year", "month"])
        cls.analyzer = PSIAnalyzer(cls.df, n_bins=5)

    def test_get_score_cuts(self):
        # Arrange
        analyzer = self.analyzer

        # Act
        cuts_df = analyzer._get_score_cuts()
        df_ = cuts_df.toPandas()
        model = df_.iloc[0]["model"]
        cuts = df_.iloc[0]["cuts"]

        # Assert
        self.assertEqual(len(df_), 1)
        self.assertEqual(model, "model1")
        self.assertEqual(len(cuts), 4)  # 4 cortes para 5 bins
        self.assertEqual(cuts, [0.1, 0.35, 0.55, 0.75])

    def test_assign_deciles_and_labels(self):
        # Arrange
        analyzer = self.analyzer
        cuts_df = analyzer._get_score_cuts()

        # Act
        df_result = analyzer._assign_deciles_and_labels(analyzer.df, cuts_df)
        result = collect_as_dict(df_result)

        # Assert
        self.assertIn("faixa_score", df_result.columns)
        self.assertIn("label_faixa_score", df_result.columns)
        self.assertTrue(all(r["faixa_score"] in [1, 2, 3, 4, 5] for r in result if r["env"] == "DEV"))

    def test_calculate_proportions_reference(self):
        # Arrange
        analyzer = self.analyzer
        cuts_df = analyzer._get_score_cuts()
        df_labeled = analyzer._assign_deciles_and_labels(analyzer.df, cuts_df)

        # Act
        df_ref = analyzer._calculate_proportions(df_labeled, is_reference=True)
        df_ref = df_ref.toPandas()
        envs = df_ref["env"].unique().tolist()

        # Assert
        self.assertIn("proporcao_ref", df_ref.columns)
        self.assertTrue(envs[0]=="DEV" and len(envs)==1)
        for p in df_ref["proporcao_ref"]:
            self.assertTrue(p==0.2)

    def test_calculate_proportions_other(self):
        # Arrange
        analyzer = self.analyzer
        cuts_df = analyzer._get_score_cuts()
        df_labeled = analyzer._assign_deciles_and_labels(analyzer.df, cuts_df)

        # Act
        df_out = analyzer._calculate_proportions(df_labeled, is_reference=False)
        df_out = df_out.toPandas()
        envs = df_out["env"].unique().tolist()

        # Assert
        self.assertIn("proporcao", df_out.columns)
        self.assertTrue(envs[0]=="PRD" and len(envs)==1)
        self.assertEqual(df_out["proporcao"].to_list(), [0.4, 0.2, 0.2, 0.2])
        

    def test_calculate_psi(self):
        # Arrange
        analyzer = self.analyzer
        cuts_df = analyzer._get_score_cuts()
        df_labeled = analyzer._assign_deciles_and_labels(analyzer.df, cuts_df)
        df_ref = analyzer._calculate_proportions(df_labeled, is_reference=True)
        df_out = analyzer._calculate_proportions(df_labeled, is_reference=False)

        # Act
        df_psi = analyzer._calculate_psi(df_out, df_ref)
        result = collect_as_dict(df_psi)

        # Assert
        self.assertEqual(df_psi.count(), 1)
        self.assertIn("psi", df_psi.columns)
        self.assertGreater(result[0]["psi"], 0)  # PSI deve ser positivo

    def test_compute_psi(self):
        # Arrange
        analyzer = self.analyzer

        # Act
        df_result = analyzer.compute_psi()
        result = collect_as_dict(df_result)

        # Assert
        self.assertEqual(df_result.count(), 1)
        self.assertIn("psi", df_result.columns)
        self.assertIn("data", df_result.columns)
        self.assertAlmostEqual(result[0]["data"].month, 2)  # mês do PRD

    # @classmethod
    # def tearDownClass(cls):
    #     cls.spark.stop()
