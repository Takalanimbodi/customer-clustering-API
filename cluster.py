# edge_functions/run_clustering.py

from fastapi import FastAPI, Request, Header, HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from joblib import load
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import os

app = FastAPI()

# Preprocessing Functions

def preprocess_aggregated(df, scaler_path="ascaler.joblib"):
    features = ['monthly_fee', 'total_revenue', 'tenure_months']

    df_proc = df[features].copy()

    df_proc['monthly_fee'] = np.log1p(df_proc['monthly_fee'])
    df_proc['total_revenue'] = np.log1p(df_proc['total_revenue'])

    scaler = load(scaler_path)

    # enforce column order + names
    df_proc = df_proc[features]
    
    scaler=load(scaler_path)
    return scaler.transform(df_proc.values)

def preprocess_behavioral(df, scaler_path="bscaler.joblib"):
    df_proc = df.copy()

    # --- membership_years engineering ---
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

    # --- feature engineering ---
    df_proc['Purchases_Per_Year'] = (
        df_proc['total_purchases'] / df_proc['membership_years'].replace(0, 1)
    ).clip(0, 500)

    df_proc['Recency_Score'] = 1 / (df_proc['days_since_last_purchase'] + 1)

    EPS = 1e-6
    df_proc['Lifetime_Value_log'] = np.log1p(df_proc['lifetime_value'] + EPS)
    df_proc['Purchases_Per_Year_log'] = np.log1p(df_proc['Purchases_Per_Year'] + EPS)

    # EXACT training feature contract
    features = [
        'Lifetime_Value_log',
        'Purchases_Per_Year_log',
        'Recency_Score'
    ]

    df_final = df_proc[features].copy()

    scaler = load(scaler_path)
    return scaler.transform(df_final.values)


from joblib import load

class AdaptiveClustererSavedModels:
    def __init__(self, agg_model_path, beh_model_path):
        self.agg_model = load(agg_model_path)
        self.beh_model = load(beh_model_path)

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


# Process New Rows Function

def process_new_rows(engine, view_name, result_table, clusterer):
    # Auto-create result table if it doesn't exist
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {result_table} (
        row_id BIGINT PRIMARY KEY,
        cluster INTEGER,
        scored_at TIMESTAMP DEFAULT NOW()
    );
    """
    with engine.connect() as conn:
        conn.execute(text(create_table_sql))
    
    # Fetch only new rows
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
    
    # Predict clusters
    df_new['cluster'] = clusterer.predict(df_new)
    
    # Store row_id and cluster to results table
    df_result = df_new[['row_id','cluster']].copy()
    df_result['scored_at'] = pd.Timestamp.now()
    
    df_result.to_sql(result_table, engine, if_exists='append', index=False, method='multi')
    
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

    clusterer = AdaptiveClustererSavedModels(
        agg_model_path="baseline_model.joblib",
        beh_model_path="behavioral_model.joblib"
    )

    # Process Aggregated Pipeline
    agg_count = process_new_rows(
        engine, 
        view_name="clean_customer_model_a", 
        result_table="clustered_results_agg", 
        clusterer=clusterer
    )

    # Process Behavioral Pipeline
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