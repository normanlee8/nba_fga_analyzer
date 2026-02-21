import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.utils import common
from prop_analyzer.data import loader

def print_accuracy_report(df, label="Total"):
    total = len(df)
    if total == 0: return

    wins = len(df[df['Result'] == 'WIN'])
    losses = len(df[df['Result'] == 'LOSS'])
    pushes = len(df[df['Result'] == 'PUSH'])
    
    decided = wins + losses
    if decided > 0:
        acc = (wins / decided) * 100
        logging.info(f"{label}: {acc:.1f}% ({wins}/{decided}) [Pushes: {pushes}]")
    else:
        logging.info(f"{label}: N/A (Only Pushes)")

def save_user_scorecard(df, date_str):
    display_cols = [
        'Player', 'Team', 'Opponent', 'Line', 
        'Expected_FGA', 'Model_Prob', 'Pick', 'Edge', 
        'Volatility', 'Confidence_Score', 'Date', 
        'Actual FGA', 'Result'
    ]
    
    final_cols = [c for c in display_cols if c in df.columns]
    if not final_cols: return

    clean_df = df[final_cols].copy()
    
    scorecard_dir = cfg.GRADED_DIR / "user_scorecards"
    scorecard_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = scorecard_dir / f"{date_str}.xlsx"
    
    try:
        def color_result(val):
            if val == 'WIN': return 'color: #008000; font-weight: bold'
            elif val == 'LOSS': return 'color: #FF0000; font-weight: bold'
            elif val == 'PUSH': return 'color: #808080; font-weight: bold'
            return ''

        try: styler = clean_df.style.map(color_result, subset=['Result'])
        except AttributeError: styler = clean_df.style.applymap(color_result, subset=['Result'])

        styler.to_excel(scorecard_path, index=False, engine='openpyxl')
        logging.info(f"Saved pretty scorecard to {scorecard_path}")
    except Exception as e:
        logging.error(f"Failed to save user scorecard: {e}")

def grade_predictions():
    preds_path = cfg.PROCESSED_OUTPUT_SYSTEM
    if not preds_path.exists():
        logging.critical(f"No predictions file found at {preds_path}")
        return

    try:
        preds_df = pd.read_parquet(preds_path)
        if preds_df.empty: return
    except Exception as e:
        logging.critical(f"Failed to load predictions: {e}")
        return

    logging.info("Loading master box scores for FGA grading...")
    full_game_df = loader.load_box_scores()

    if full_game_df is None or full_game_df.empty:
        logging.warning("No master box scores found. Cannot grade props.")
        return

    logging.info(f"Grading {len(preds_df)} FGA predictions...")

    if Cols.DATE in full_game_df.columns:
        full_game_df[Cols.DATE] = pd.to_datetime(full_game_df[Cols.DATE]).dt.normalize()

    preds_df['Match_Date'] = pd.to_datetime(preds_df['Date']).dt.normalize()
    
    results = []
    
    for idx, row in preds_df.iterrows():
        p_date = row['Match_Date']
        p_name = str(row.get('Player', '')).lower().strip()
        
        # Smart Matching: Look for the player's game within +/- 2 days
        match = pd.DataFrame()
        if 'PLAYER_NAME' in full_game_df.columns:
            mask = full_game_df['PLAYER_NAME'].str.lower().str.strip() == p_name
            player_games = full_game_df[mask].copy()
            
            if not player_games.empty:
                # Calculate date difference in days
                player_games['Days_Diff'] = (pd.to_datetime(player_games[Cols.DATE]) - p_date).dt.days.abs()
                # Filter to games played within 2 days of the prediction date
                valid_games = player_games[player_games['Days_Diff'] <= 2].sort_values('Days_Diff')
                
                if not valid_games.empty:
                    match = valid_games.iloc[[0]] 
        
        if match.empty:
            row['Actual FGA'] = None
            row['Result'] = 'Pending / Not Found'
            results.append(row)
            continue
            
        actual_fga = match.iloc[0].get('FGA')
        
        if pd.isna(actual_fga):
            row['Actual FGA'] = None
            row['Result'] = 'Stat is NaN'
            results.append(row)
            continue
            
        row['Actual FGA'] = actual_fga
        
        try:
            line = float(row['Line'])
            pick = row['Pick']
            
            if pick == 'Over': res = 'WIN' if actual_fga > line else 'LOSS' if actual_fga < line else 'PUSH'
            elif pick == 'Under': res = 'WIN' if actual_fga < line else 'LOSS' if actual_fga > line else 'PUSH'
            else: res = 'ERROR'
        except Exception:
            res = 'Error Grading'
            
        row['Result'] = res
        results.append(row)

    graded_df = pd.DataFrame(results)
    
    logging.info("-" * 40)
    logging.info(">>> FGA GRADING REPORT <<<")
    
    finished = graded_df[graded_df['Result'].isin(['WIN', 'LOSS', 'PUSH'])].copy()
    print_accuracy_report(finished, "Total FGA Props")
    
    if 'Confidence_Score' in finished.columns:
        high_conf = finished[finished['Confidence_Score'] >= 15.0]
        med_conf = finished[(finished['Confidence_Score'] >= 10.0) & (finished['Confidence_Score'] < 15.0)]
        low_conf = finished[finished['Confidence_Score'] < 10.0]
        
        print_accuracy_report(high_conf, "High Confidence (> 15.0)")
        print_accuracy_report(med_conf, "Solid Confidence (10.0 - 15.0)")
        print_accuracy_report(low_conf, "Low Confidence (< 10.0)")

    logging.info("-" * 40)
    
    system_history_df = graded_df.copy()
    system_history_df[Cols.ACTUAL_VAL] = system_history_df['Actual FGA']
    system_history_df[Cols.PREDICTION] = system_history_df['Expected_FGA']
    system_history_df[Cols.PROP_TYPE] = 'FGA'
    system_history_df[Cols.PLAYER_NAME] = system_history_df['Player']
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    parquet_path = cfg.GRADED_DIR / f"graded_props_{today_str}.parquet"
    
    try:
        for col in system_history_df.select_dtypes(include=['object']).columns:
            system_history_df[col] = system_history_df[col].astype(str)
            
        system_history_df.to_parquet(parquet_path, index=False)
        save_user_scorecard(graded_df, today_str)
        logging.info(f"Saved graded FGA results for {today_str}")
        
    except Exception as e:
        logging.error(f"Failed to save output: {e}")

def main():
    common.setup_logging(name="fga_grading")
    grade_predictions()

if __name__ == "__main__":
    main()