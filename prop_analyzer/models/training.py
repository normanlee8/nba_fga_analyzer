import sys
import pandas as pd
import numpy as np
import logging
import xgboost as xgb
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.models import registry
from prop_analyzer.utils import common

# Constants
TEST_SET_SIZE_PCT = 0.20

def calculate_time_decay_weights(df, date_col, decay_rate=0.015):
    """Calculates sample weights based on exponential time decay."""
    if date_col not in df.columns:
        return pd.Series(1.0, index=df.index)
        
    dates = pd.to_datetime(df[date_col])
    max_date = dates.max()
    days_diff = (max_date - dates).dt.days
    return np.exp(-decay_rate * days_diff)

def add_interaction_features(df):
    """Creates interaction features to help find correlation between vacancy and FGA faster."""
    if 'Primary_Pos' in df.columns:
        is_guard = df['Primary_Pos'].astype(str).str.contains('G', case=False, regex=True).astype(int)
        is_forward = df['Primary_Pos'].astype(str).str.contains('F', case=False, regex=True).astype(int)
        
        if 'MISSING_USG_G' in df.columns:
            df['INT_GUARD_VACANCY'] = is_guard * df['MISSING_USG_G']
            
        if 'MISSING_USG_F' in df.columns:
            df['INT_FORWARD_VACANCY'] = is_forward * df['MISSING_USG_F']
            
    return df

def generate_fga_synthetic_lines(df):
    """Generates synthetic FGA lines if historical lines are missing."""
    n_rows = len(df)
    
    szn = df.get('FGA_SZN_AVG', df['FGA']).fillna(0)
    l5 = df.get('FGA_L5_AVG', szn).fillna(szn)
    l10 = df.get('FGA_L10_AVG', l5).fillna(l5)
    
    w_szn = np.random.uniform(0.4, 0.8, size=n_rows)
    w_recent = 1.0 - w_szn
    recent_mix = (0.6 * l5) + (0.4 * l10)
    
    base_proj = (w_szn * szn) + (w_recent * recent_mix)
    
    if 'TEAM_PACE' in df.columns:
        pace_factor = df['TEAM_PACE'].fillna(100.0) / 100.0
        base_proj = base_proj * np.sqrt(pace_factor)
        
    market_noise = np.random.normal(0, 0.6, size=n_rows)
    raw_line = base_proj + market_noise
    final_line = np.maximum(np.round(raw_line * 2) / 2, 0.5) 
    
    return final_line

def get_fga_feature_cols(all_columns):
    """Hardcoded list of specifically engineered FGA Volume Drivers."""
    core_features = [
        'FGA_SZN_AVG', 'FGA_L5_AVG', 'FGA_L10_AVG',
        'FGA_L5_STD', 'FGA_L10_STD', 'FGA_L20_STD',
        'SZN_USG_PROXY', 'L5_USG_PROXY',
        'TEAM_MISSING_USG', 'MISSING_USG_G', 'MISSING_USG_F', 'MISSING_USG_C',
        'INT_GUARD_VACANCY', 'INT_FORWARD_VACANCY',
        'TEAM_PACE', 'OPP_PACE'
    ]
    
    defensive_cols = [c for c in all_columns if 'OPP_' in c and 'FGA' in c]
    core_features.extend(defensive_cols)
    
    return [c for c in core_features if c in all_columns]

def train_fga_model(df):
    logging.info("Training FGA Optimization Models...")
    
    target_col = 'FGA'
    if target_col not in df.columns:
        logging.error("Target column 'FGA' is missing from the dataset.")
        return

    # --- TIME SERIES SPLIT PROTECTION ---
    date_col = Cols.DATE if Cols.DATE in df.columns else 'GAME_DATE'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col, ascending=True).reset_index(drop=True)

    df = add_interaction_features(df)

    # --- SYNTHETIC LINE GENERATION ---
    if Cols.PROP_LINE not in df.columns or df[Cols.PROP_LINE].sum() == 0:
        logging.info("Generating SMART synthetic FGA lines...")
        df[Cols.PROP_LINE] = generate_fga_synthetic_lines(df)
    
    df = df.dropna(subset=[Cols.PROP_LINE, target_col]).copy()

    # --- SAMPLE WEIGHT CALCULATION ---
    sample_weights = calculate_time_decay_weights(df, date_col)

    # Select Features
    feature_list = get_fga_feature_cols(df.columns)
    sanitized_cols = [re.sub(r'[^\w\s]', '_', str(col)).replace(' ', '_') for col in feature_list]
    
    X = df[feature_list].fillna(0).copy()
    X.columns = sanitized_cols
    y = df[target_col]
    
    split_idx = int(len(X) * (1 - TEST_SET_SIZE_PCT))
    
    # SPLIT DATA
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    w_train = sample_weights.iloc[:split_idx]
    
    line_train = df.iloc[:split_idx][Cols.PROP_LINE]
    line_val = df.iloc[split_idx:][Cols.PROP_LINE]
    
    y_train_clf = (y_train > line_train).astype(int)
    y_val_clf = (y_val > line_val).astype(int)

    preprocessor = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    # --- MODEL 1: FGA REGRESSOR (Optimized for Mean Absolute Error) ---
    logging.info("Fitting XGBoost FGA Regressor (MAE Objective)...")
    regressor = xgb.XGBRegressor(
        objective='reg:absoluteerror', 
        n_estimators=750, 
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8, 
        colsample_bytree=0.8
    )
    regressor.fit(
        X_train_proc, y_train, 
        sample_weight=w_train, 
        eval_set=[(X_val_proc, y_val)], 
        verbose=False
    )
    
    val_preds = regressor.predict(X_val_proc)
    mae = mean_absolute_error(y_val, val_preds)
    logging.info(f"[FGA] Validation Mean Absolute Error: {mae:.2f} Shot Attempts")

    # --- MODEL 2: FGA CLASSIFIER (Win Probability) ---
    logging.info("Fitting XGBoost FGA Classifier (Probability Objective)...")
    classifier = xgb.XGBClassifier(
        objective='binary:logistic', 
        n_estimators=600, 
        learning_rate=0.03, 
        eval_metric='logloss',
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8
    )
    classifier.fit(
        X_train_proc, y_train_clf, 
        sample_weight=w_train, 
        eval_set=[(X_val_proc, y_val_clf)], 
        verbose=False
    )

    clf_preds = classifier.predict(X_val_proc)
    acc = accuracy_score(y_val_clf, clf_preds)
    logging.info(f"[FGA] Validation Over/Under Accuracy: {acc:.1%}")

    artifacts = {
        'scaler': preprocessor,
        'features': sanitized_cols,
        'regressor': regressor,
        'classifier': classifier
    }
    registry.save_artifacts('FGA', artifacts)

def main():
    common.setup_logging(name="train_fga_model")
    logging.info(">>> STARTING FGA MODEL TRAINING PIPELINE")

    train_file = cfg.MASTER_TRAINING_FILE
    if not train_file.exists():
        logging.critical(f"Training dataset not found at {train_file}")
        return

    try:
        df = pd.read_parquet(train_file)
        if df.empty:
            logging.critical("Training dataset is empty.")
            return
            
        logging.info(f"Loaded {len(df)} rows of training data.")
        train_fga_model(df)
        
    except Exception as e:
        logging.critical(f"Failed to execute training pipeline: {e}")

if __name__ == "__main__":
    main()