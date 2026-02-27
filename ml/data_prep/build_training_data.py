"""
Pull Gold-layer hazard risk data from Athena, split into train/test,
and save as Parquet files for ML training.

Gold layer source: s3://hazard/gold/risk_feature_mart/
"""
import json
import logging
from pathlib import Path

import awswrangler as wr
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "model_config.yml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def pull_gold_data(database: str, table: str) -> pd.DataFrame:
    """Query the Gold-layer risk feature mart from Athena."""
    logger.info("Querying Athena: %s.%s", database, table)
    query = f"""
        SELECT *
        FROM {database}.{table}
        WHERE NRI_ExpectedLoss IS NOT NULL
        ORDER BY county_fips, year
    """
    df = wr.athena.read_sql_query(
        sql=query,
        database=database,
        ctas_approach=False,
    )
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def add_risk_bucket(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Bin the continuous target into LOW/MEDIUM/HIGH for stratified splitting."""
    df = df.copy()
    df["risk_bucket"] = pd.qcut(
        df[target_col],
        q=3,
        labels=["LOW", "MEDIUM", "HIGH"],
        duplicates="drop",
    )
    return df


def split_and_save(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    stratify_col: str,
    train_ratio: float,
    output_dir: str,
    random_state: int = 42,
) -> tuple:
    """Stratified train/test split and persist to Parquet (local or S3)."""
    cols = feature_cols + [target_col, stratify_col]
    cols = [c for c in cols if c in df.columns]
    df_clean = df[cols].dropna(subset=[target_col])

    train_df, test_df = train_test_split(
        df_clean,
        train_size=train_ratio,
        stratify=df_clean[stratify_col],
        random_state=random_state,
    )
    logger.info("Train: %d rows | Test: %d rows", len(train_df), len(test_df))

    is_s3 = output_dir.startswith("s3://")
    if is_s3:
        wr.s3.to_parquet(train_df, path=f"{output_dir}train.parquet")
        wr.s3.to_parquet(test_df, path=f"{output_dir}test.parquet")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
        test_df.to_parquet(f"{output_dir}/test.parquet", index=False)

    logger.info("Saved train/test Parquet to %s", output_dir)
    return train_df, test_df


def save_feature_list(feature_cols: list, target_col: str, output_dir: str) -> None:
    """Persist the feature schema for downstream use (inference, monitoring)."""
    schema = {"features": feature_cols, "target": target_col}
    is_s3 = output_dir.startswith("s3://")
    if is_s3:
        wr.s3.to_json(
            pd.DataFrame([schema]),
            path=f"{output_dir}feature_list.json",
        )
    else:
        with open(f"{output_dir}/feature_list.json", "w") as f:
            json.dump(schema, f, indent=2)
    logger.info("Saved feature_list.json")


def run(output_dir=None):
    config = load_config()
    data_cfg = config["data"]
    feat_cfg = config["features"]

    df = pull_gold_data(data_cfg["database"], data_cfg["table"])
    df = add_risk_bucket(df, data_cfg["target_column"])

    feature_cols = feat_cfg["numeric"] + feat_cfg.get("categorical", [])
    feature_cols = [c for c in feature_cols if c in df.columns]

    out = output_dir or data_cfg["output_dir"]
    train_df, test_df = split_and_save(
        df=df,
        feature_cols=feature_cols,
        target_col=data_cfg["target_column"],
        stratify_col=data_cfg["stratify_column"],
        train_ratio=data_cfg["train_ratio"],
        output_dir=out,
        random_state=config["model"].get("random_state", 42),
    )
    save_feature_list(feature_cols, data_cfg["target_column"], out)
    return train_df, test_df


if __name__ == "__main__":
    run()
