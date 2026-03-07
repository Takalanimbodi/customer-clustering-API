from fastapi import FastAPI, Request, Header, HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from joblib import load
import pandas as pd
import numpy as np
import os

app = FastAPI()

# Absolute path to models folder (prevents Render path issues)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")



# Preprocessing Functions

def preprocess_aggregated(df):

    features = ['monthly_fee', 'total_revenue', 'tenure_months']
    df_proc = df[features].copy()

    df_proc['monthly_fee'] = np.log1p(df_proc['monthly_fee'])
    df_proc['total_revenue'] = np.log1p(df_proc['total_revenue'])

    scaler = load(os.path.join(MODELS_DIR, "ascaler.joblib"))

    df_proc = df_proc[features]

    return scaler.transform(df_proc.values)


def preprocess_behavioral(df):

    df_proc = df.copy()

    if 'membership_years' not in df_proc.columns:
        if 'membership_months' in df_proc.columns:
            df_proc['membership_years'] = df_proc['membership_months'] / 12
        else:
            mask_active = (
                (df_proc.get('membership', 0) == 1) &
                (
                    (df_proc.get('total_purchases', 0) > 0) |
                    (df_proc.get('tenure_months', 0) > 0)
                )
            )
            df_proc.loc[mask_active, 'membership_years'] = 1

    df_proc = df_proc.dropna(subset=['membership_years'])

    df_proc['Purchases_Per_Year'] = (
        df_proc['total_purchases'] / df_proc['membership_years'].replace(0, 1)
    ).clip(0, 500)

    df_proc['Recency_Score'] = 1 / (df_proc['days_since_last_purchase'] + 1)

    EPS = 1e-6
    df_proc['Lifetime_Value_log'] = np.log1p(df_proc['lifetime_value'] + EPS)
    df_proc['Purchases_Per_Year_log'] = np.log1p(df_proc['Purchases_Per_Year'] + EPS)

    features = [
        'Lifetime_Value_log',
        'Purchases_Per_Year_log',
        'Recency_Score'
    ]

    df_final = df_proc[features].copy()

    scaler = load(os.path.join(MODELS_DIR, "bscaler.joblib"))

    return scaler.transform(df_final.values)


# Cluster Model Loader

class AdaptiveClustererSavedModels:

    def __init__(self):

        self.agg_model = load(os.path.join(MODELS_DIR, "baseline_model.joblib"))
        self.beh_model = load(os.path.join(MODELS_DIR, "behavioral_model.joblib"))

    def detect_model(self, df):

        beh_raw_cols = {
            'total_purchases',
            'lifetime_value',
            'days_since_last_purchase'
        }

        agg_cols = {
            'tenure_months',
            'monthly_fee',
            'total_revenue'
        }

        if beh_raw_cols.issubset(df.columns):
            return 'behavioral'
        elif agg_cols.issubset(df.columns):
            return 'aggregated'
        else:
            raise ValueError(
                f"Dataset columns do not match expected schemas: {df.columns.tolist()}"
            )

    def predict(self, df_new):

        model_type = self.detect_model(df_new)

        if model_type == 'behavioral':
            X_scaled = preprocess_behavioral(df_new)
            return self.beh_model.predict(X_scaled)
        else:
            X_scaled = preprocess_aggregated(df_new)
            return self.agg_model.predict(X_scaled)



# Process New Rows

def process_new_rows(engine, view_name, result_table, clusterer):

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {result_table} (
        row_id BIGINT PRIMARY KEY,
        cluster INTEGER,
        scored_at TIMESTAMP DEFAULT NOW()
    );
    """

    with engine.connect() as conn:
        conn.execute(text(create_table_sql))

    query = f"""
        SELECT * FROM {view_name} v
        WHERE NOT EXISTS (
            SELECT 1 FROM {result_table} r
            WHERE r.row_id = v.row_id
        )
    """

    df_new = pd.read_sql(query, engine)

    if df_new.empty:
        return 0

    df_new['cluster'] = clusterer.predict(df_new)

    df_result = df_new[['row_id','cluster']].copy()
    df_result['scored_at'] = pd.Timestamp.now()

    df_result.to_sql(
        result_table,
        engine,
        if_exists='append',
        index=False,
        method='multi'
    )

    return len(df_new)



# FastAPI Endpoint

@app.post("/run_clustering")
async def run_clustering(request: Request, x_api_secret: str = Header(...)):

    if x_api_secret != os.environ["API_SECRET"]:
        raise HTTPException(status_code=401, detail="Unauthorized")

    engine = create_engine(
        os.environ["DATABASE_URL"],
        poolclass=NullPool
    )

    clusterer = AdaptiveClustererSavedModels()

    agg_count = process_new_rows(
        engine,
        view_name="clean_customer_model_a",
        result_table="clustered_results_agg",
        clusterer=clusterer
    )

    beh_count = process_new_rows(
        engine,
        view_name="clean_customer_model_b",
        result_table="clustered_results_beh",
        clusterer=clusterer
    )

    return {
        "status": "success",
        "aggregated_rows_clustered": agg_count,
        "behavioral_rows_clustered": beh_count
    }