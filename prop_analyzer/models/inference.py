import sys
import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from datetime import timedelta

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.models import registry
from prop_analyzer.features.calculator import calculate_blowout_risk_multiplier
from prop_analyzer.models.training import add_interaction_features

def load_artifacts():
    return registry.load_artifacts('FGA')

def get_recent_bias_map(days_back=21):
    """
    Loads graded history and calculates the average model FGA error per player.
    Positive Bias = Model Undershoots -> We add to projection.
    Negative Bias = Model Overshoots -> We subtract from projection.
    """
    graded_files = sorted(cfg.GRADED_DIR.glob("graded_*.parquet"), reverse=True)
    if not graded_files:
        return {}

    recent_dfs = []
    cutoff_date = pd.Timestamp.now() - timedelta(days=days_back)
    
    for f in graded_files:
        try:
            file_date_str = f.stem.replace('graded_props_', '').replace('graded_', '')
            file_date = pd.to_datetime(file_date_str)
            if file_date < cutoff_date:
                continue
                
            df = pd.read_parquet(f)
            # Filter strictly for FGA
            df = df[df[Cols.PROP_TYPE] == 'FGA']
            
            if Cols.ACTUAL_VAL in df.columns and Cols.PREDICTION in df.columns:
                keep_cols = [Cols.PLAYER_ID, Cols.PLAYER_NAME, Cols.ACTUAL_VAL, Cols.PREDICTION]
                recent_dfs.append(df[[c for c in keep_cols if c in df.columns]].copy())
        except Exception:
            continue
            
    if not recent_dfs:
        return {}
        
    full_history = pd.concat(recent_dfs, ignore_index=True)
    full_history[Cols.ACTUAL_VAL] = pd.to_numeric(full_history[Cols.ACTUAL_VAL], errors='coerce')
    full_history[Cols.PREDICTION] = pd.to_numeric(full_history[Cols.PREDICTION], errors='coerce')
    full_history.dropna(subset=[Cols.ACTUAL_VAL, Cols.PREDICTION], inplace=True)
    
    full_history['Error'] = full_history[Cols.ACTUAL_VAL] - full_history[Cols.PREDICTION]
    
    group_col = Cols.PLAYER_ID if Cols.PLAYER_ID in full_history.columns else Cols.PLAYER_NAME
    bias_series = full_history.groupby(group_col)['Error'].mean()
    
    return bias_series.to_dict()

def predict_props(todays_props_df):
    logging.info(f"Starting FGA inference for {len(todays_props_df)} props...")
    
    # 1. Load Artifacts
    artifacts = load_artifacts()
    if not artifacts:
        logging.critical("FGA Artifacts missing! Run training.py first.")
        return pd.DataFrame()
        
    classifier = artifacts['classifier']
    regressor = artifacts['regressor']
    scaler = artifacts['scaler']
    feature_names = artifacts['features']
    
    # 2. Load Continuous Learning Bias Map
    try:
        bias_map = get_recent_bias_map(days_back=21)
        logging.info(f"Loaded FGA bias corrections for {len(bias_map)} players.")
    except Exception as e:
        logging.warning(f"Failed to load bias map: {e}")
        bias_map = {}

    # 3. Feature Preparation
    X_raw = todays_props_df.copy()
    X_raw = add_interaction_features(X_raw)
    
    X_model = pd.DataFrame(index=X_raw.index)
    sanitized_map = {c: re.sub(r'[^\w\s]', '_', str(c)).replace(' ', '_') for c in X_raw.columns}
    inv_map = {v: k for k, v in sanitized_map.items()}
    
    for f in feature_names:
        if f in X_raw.columns:
            X_model[f] = X_raw[f]
        elif f in inv_map:
            X_model[f] = X_raw[inv_map[f]]
        else:
            X_model[f] = 0.0 

    # 4. Predict
    try:
        X_scaled = scaler.transform(X_model)
        raw_fga_preds = regressor.predict(X_scaled)
        over_probs = classifier.predict_proba(X_scaled)[:, 1]
    except Exception as e:
        logging.error(f"Failed during model prediction: {e}")
        return pd.DataFrame()

    # 5. Post-Processing (Blowout Risk & Confidence Math)
    results = []
    
    for idx, (orig_idx, row) in enumerate(X_raw.iterrows()):
        line = float(row.get(Cols.PROP_LINE, 0.0))
        if line == 0: continue
            
        prob_over = float(over_probs[idx])
        raw_proj = float(raw_fga_preds[idx])
        
        # Calculate Blowout Haircut
        team_margin = float(row.get('TEAM_Average Scoring Margin', 0.0))
        opp_margin = float(row.get('OPP_Average Scoring Margin', 0.0))
        blowout_mult = calculate_blowout_risk_multiplier(team_margin, opp_margin)
        
        # Apply Bias Correction
        p_id = row.get(Cols.PLAYER_ID)
        key = int(p_id) if not pd.isna(p_id) else row.get(Cols.PLAYER_NAME)
        bias = bias_map.get(key, 0.0)
        
        # Final Expected FGA
        mu_fga = (raw_proj * blowout_mult) + (bias * 0.5)
        
        # FGA Volatility for Confidence Algorithm
        # Fallback to 1.5 shots if standard deviation is missing
        volatility = float(row.get('FGA_L10_STD', 1.5))
        if pd.isna(volatility) or volatility <= 0:
            volatility = 1.5
            
        # The Math
        edge = mu_fga - line
        confidence_score = (abs(edge) / volatility) * 100.0
        
        is_over = mu_fga > line
        pick = 'Over' if is_over else 'Under'
        win_prob = prob_over if is_over else (1.0 - prob_over)

        res_dict = {
            Cols.PLAYER_NAME: row[Cols.PLAYER_NAME],
            Cols.TEAM: row.get('TEAM_ABBREVIATION', row.get(Cols.TEAM, 'UNK')),
            Cols.OPPONENT: row.get(Cols.OPPONENT, 'UNK'),
            Cols.DATE: row[Cols.DATE],
            Cols.PROP_TYPE: 'FGA',
            Cols.PROP_LINE: line,
            'Expected_FGA': round(mu_fga, 2),
            'Model_Prob': round(win_prob, 3),
            'Pick': pick,
            'Edge': round(edge, 2),
            'Volatility': round(volatility, 2),
            'Confidence_Score': round(confidence_score, 1)
        }
        results.append(res_dict)

    final_df = pd.DataFrame(results)
    if final_df.empty:
        return pd.DataFrame()
        
    # Strictly sort by the mathematical Confidence Score descending
    final_df = final_df.sort_values(by='Confidence_Score', ascending=False).reset_index(drop=True)
    
    return final_df