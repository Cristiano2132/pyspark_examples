import unittest
from pyspark.sql import SparkSession, functions as F
from faixas_score_calculator import FaixaScoreCalculator 


class TestFaixaScoreCalculator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[1]").appName("TestFaixaScore").getOrCreate()
        cls.spark.sparkContext.setLogLevel("ERROR")

        # Simulação dos dados
        data = []
        n = 100
        for modelo in ["modelo_1", "modelo_2"]:
            for i in range(n):
                score = i % 10 + 1  # 10 faixas de score
                valor = 1 if i % 2 == 0 else 0
                data.append(("2024-01", score, valor, modelo, "dev"))
                data.append(("2024-02", 10 - score, valor, modelo, "oot"))
                data.append(("2024-03", 5 if i < n // 2 else 9, valor, modelo, "oot"))

        cls.df = cls.spark.createDataFrame(data, ["periodo", "score", "valor_observado", "modelo_id", "ambiente"])

    def setUp(self):
        self.calculator = FaixaScoreCalculator(
            df=self.df,
            ano_mes_column="periodo",
            vr_column="valor_observado",
            score_column="score",
            modelo_column="modelo_id",
            ambiente_column="ambiente",
            dev_code="dev"
        )
        self.n_faixas_score = 2
        self.calculator.set_cuts(self.n_faixas_score)
        
    def test_create_instance(self):
        self.assertIsInstance(self.calculator, FaixaScoreCalculator)


    def test_must_set_cuts(self):
        self.assertIsNotNone(self.calculator.cuts)
        
    def test_get_faixas_score_dev(self):
        self.calculator._get_faixas_score_dev()
        self.assertIsNotNone(self.calculator.df)
        self.assertIn("faixa_score", self.calculator.df.columns)
        self.assertIn("faixa_score_label", self.calculator.df.columns)

    def test_get_faixas_score(self):
        spkdf_faixas_score = self.calculator.get_faixas_score()
        self.assertIsNotNone(spkdf_faixas_score)
        self.assertIn("count_por_faixa", spkdf_faixas_score.columns)
        self.assertIn("eventos_por_faixa", spkdf_faixas_score.columns)
        self.assertIn("risco_faixa", spkdf_faixas_score.columns)
        self.assertIn("taxa_eventos_por_faixa", spkdf_faixas_score.columns)
        self.assertIn("taxa_obs_por_faixa", spkdf_faixas_score.columns)
        self.assertIn("modelo_id", spkdf_faixas_score.columns)
        self.assertIn("ambiente", spkdf_faixas_score.columns)
        self.assertIn("periodo", spkdf_faixas_score.columns)

    def test_distribuicao_dev_igualitaria(self):
        spkdf_faixas_score = self.calculator.get_faixas_score()
        dev_data = spkdf_faixas_score.filter(F.col("ambiente") == "dev")

        counts = dev_data.select("faixa_score", "count_por_faixa").groupBy("faixa_score").agg(
            F.sum("count_por_faixa").alias("total")
        ).collect()

        valores = [row["total"] for row in counts]
        self.assertTrue(all(val == valores[0] for val in valores), "Distribuição não está igualitária em DEV")

    def test_must_return_not_empty_dataframe(self):
        spkdf_faixas_score = self.calculator.get_faixas_score()
        n = spkdf_faixas_score.count()
        n_esperado = self.n_faixas_score * 3 * 2
        self.assertEqual(n, n_esperado, "Número de linhas não está igual ao esperado")
    
    def test_sum_of_taxa_obs_por_faixa_must_be_1(self):
        spkdf_faixas_score = self.calculator.get_faixas_score()
        spkdf_faixas_score_agg = spkdf_faixas_score\
            .groupBy("modelo_id", "ambiente", "periodo")\
                .agg(
                    F.sum("taxa_obs_por_faixa").alias("sum_taxa_obs_por_faixa")
                )
        list_sum_taxa_obs_por_faixa = spkdf_faixas_score_agg\
            .select("sum_taxa_obs_por_faixa")\
            .rdd.flatMap(lambda x: x).collect()
        self.assertTrue(all(val == 1 for val in list_sum_taxa_obs_por_faixa), "Soma das taxas não é igual a 1")

    def test_sum_of_count_por_faixa_must_be_equal_to_count_por_periodo(self):
        spkdf_faixas_score = self.calculator.get_faixas_score()
        
        spkdf_faixas_score_agg = spkdf_faixas_score\
            .groupBy("modelo_id", "ambiente", "periodo")\
            .agg(
                F.sum("count_por_faixa").alias("sum_count_por_faixa"),
                F.mean("count_por_periodo").alias("mean_count_por_periodo")
            )
        
        list_sum_count_por_faixa = spkdf_faixas_score_agg\
            .select("sum_count_por_faixa")\
            .rdd.flatMap(lambda x: x).collect()
        list_sum_count_por_periodo = spkdf_faixas_score_agg\
            .select("mean_count_por_periodo")\
            .rdd.flatMap(lambda x: x).collect()
        self.assertEqual(list_sum_count_por_faixa, list_sum_count_por_periodo)

    def test_risco_e_taxas_com_valores_controlados(self):
        # Simula um DataFrame menor com valores fáceis de validar
        data = [
            ("2024-01", 0.1, 1, "modelo_1", "dev"),
            ("2024-01", 0.2, 0, "modelo_1", "dev"),
            ("2024-01", 0.8, 1, "modelo_1", "dev"),
            ("2024-01", 0.9, 1, "modelo_1", "dev"),
        ]
        df_teste = self.spark.createDataFrame(data, ["periodo", "score", "valor_observado", "modelo_id", "ambiente"])

        calc = FaixaScoreCalculator(
            df=df_teste,
            ano_mes_column="periodo",
            vr_column="valor_observado",
            score_column="score",
            modelo_column="modelo_id",
            ambiente_column="ambiente",
            dev_code="dev"
        )
        # 2 faixas -> corte em 0.5
        calc.set_cuts(2)

        resultado = calc.get_faixas_score().filter(F.col("ambiente") == "dev").collect()
        resultado = sorted(resultado, key=lambda row: row["faixa_score"])

        faixa_1 = resultado[0]
        faixa_2 = resultado[1]

        # Faixa 1: scores 0.1, 0.2 → eventos: 1 + 0 = 1 → risco = 0.5
        # Faixa 2: scores 0.8, 0.9 → eventos: 1 + 1 = 2 → risco = 1.0

        self.assertEqual(faixa_1["count_por_faixa"], 2)
        self.assertEqual(faixa_1["eventos_por_faixa"], 1)
        self.assertAlmostEqual(faixa_1["risco_faixa"], 0.5, places=2)
        self.assertAlmostEqual(faixa_1["taxa_eventos_por_faixa"], 1 / 4, places=2)
        self.assertAlmostEqual(faixa_1["taxa_obs_por_faixa"], 0.5, places=2)

        self.assertEqual(faixa_2["count_por_faixa"], 2)
        self.assertEqual(faixa_2["eventos_por_faixa"], 2)
        self.assertAlmostEqual(faixa_2["risco_faixa"], 1.0, places=2)
        self.assertAlmostEqual(faixa_2["taxa_eventos_por_faixa"], 2 / 4, places=2)
        self.assertAlmostEqual(faixa_2["taxa_obs_por_faixa"], 0.5, places=2)