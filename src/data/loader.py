from pathlib import Path
from typing import Optional
import pandas as pd
from src.core.config import settings
from src.core.logging import logger
from src.data.preprocessor import TextCleanerPTBR


class OlistDataLoader:
    """Classe responsável pelo carregamento, junção relacional e preparação dos dados da Olist."""

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ):
        self.raw_dir = raw_dir or settings.DATA_RAW_DIR
        self.processed_dir = processed_dir or settings.DATA_PROCESSED_DIR

    def load_raw_reviews(self) -> pd.DataFrame:
        """Carrega o ficheiro de avaliações dos clientes."""
        reviews_path = self.raw_dir / "olist_order_reviews_dataset.csv"
        if not reviews_path.exists():
            raise FileNotFoundError(
                f"Ficheiro não encontrado: {reviews_path}. "
                "Transfira a base de dados da Olist do Kaggle para data/raw/."
            )
        logger.info(f"A carregar avaliações a partir de: {reviews_path}")
        return pd.read_csv(reviews_path)

    def load_raw_orders(self) -> pd.DataFrame:
        """Carrega o ficheiro de encomendas para enriquecimento de metadados de entrega."""
        orders_path = self.raw_dir / "olist_orders_dataset.csv"
        if not orders_path.exists():
            logger.warning(
                f"Ficheiro de encomendas não encontrado em {orders_path}. "
                "O enriquecimento temporal não será executado."
            )
            return pd.DataFrame()
        logger.info(f"A carregar encomendas a partir de: {orders_path}")
        return pd.read_csv(
            orders_path,
            parse_dates=[
                "order_purchase_timestamp",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        )

    def process_and_enrich(self) -> pd.DataFrame:
        """Executa a limpeza, cálculo de atrasos e consolidação dos comentários."""
        df_reviews = self.load_raw_reviews()
        df_orders = self.load_raw_orders()

        logger.info(f"Total de registos brutos de avaliações: {len(df_reviews):,}")

        # 1. Filtro: Manter apenas registos que possuam comentário em texto
        df_reviews["review_comment_message"] = df_reviews["review_comment_message"].fillna("")
        df_reviews["review_comment_title"] = df_reviews["review_comment_title"].fillna("")

        # Combinar título com mensagem para criar o texto completo
        df_reviews["full_comment"] = (
            df_reviews["review_comment_title"] + " " + df_reviews["review_comment_message"]
        ).str.strip()

        # Filtrar apenas registos com texto útil após união
        df_valid = df_reviews[df_reviews["full_comment"].str.len() > 3].copy()
        logger.info(f"Avaliações com texto relevante: {len(df_valid):,}")

        # 2. Enriquecimento de Metadados Temporais (se df_orders estiver presente)
        if not df_orders.empty:
            df_merged = df_valid.merge(
                df_orders[[
                    "order_id",
                    "order_status",
                    "order_purchase_timestamp",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                ]],
                on="order_id",
                how="left",
            )

            # Cálculo do atraso na entrega em dias (positivo = atraso; negativo = adiantado)
            diff_days = (
                df_merged["order_delivered_customer_date"] - df_merged["order_estimated_delivery_date"]
            ).dt.total_seconds() / 86400.0

            df_merged["delivery_delay_days"] = diff_days.round(1)
            df_merged["is_delayed"] = df_merged["delivery_delay_days"] > 0
        else:
            df_merged = df_valid
            df_merged["delivery_delay_days"] = None
            df_merged["is_delayed"] = None

        # 3. Limpeza Textual com o sanitizador PT-BR
        logger.info("A aplicar limpeza textual adaptada ao Português...")
        df_merged["clean_comment"] = df_merged["full_comment"].apply(TextCleanerPTBR.clean)

        # Filtrar novamente caso a limpeza tenha esvaziado o conteúdo
        df_final = df_merged[df_merged["clean_comment"].str.len() > 2].copy()

        # Seleccionar e ordenar colunas finais para o índice do RAG
        columns_to_keep = [
            "review_id",
            "order_id",
            "review_score",
            "clean_comment",
            "full_comment",
            "review_creation_date",
            "delivery_delay_days",
            "is_delayed",
        ]
        available_columns = [col for col in columns_to_keep if col in df_final.columns]
        df_final = df_final[available_columns]

        logger.info(f"Base processada finalizada com sucesso: {len(df_final):,} registos.")
        return df_final

    def save_processed(self, df: pd.DataFrame, file_name: str = "olist_reviews_clean.parquet") -> Path:
        """Exporta o DataFrame consolidado para ficheiro Parquet."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_dir / file_name
        df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info(f"Base de dados gravada em: {output_path}")
        return output_path


if __name__ == "__main__":
    loader = OlistDataLoader()
    df_clean = loader.process_and_enrich()
    loader.save_processed(df_clean)