import pandas as pd
import logging
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.data import loader
from prop_analyzer.features import generator

def create_training_dataset():
    """
    Builds the final FGA training dataset.
    Combines Master Box Scores, Real Vegas Lines, and FGA Rolling Features.
    """
    logging.info("--- Building Final FGA Training Dataset ---")
    
    # 1. Load Base Box Scores
    box_scores = loader.load_box_scores()
    if box_scores is None or box_scores.empty:
        logging.error("No box scores available. Cannot build training set.")
        return

    if Cols.DATE in box_scores.columns:
        box_scores[Cols.DATE] = pd.to_datetime(box_scores[Cols.DATE]).dt.normalize()

    # 2. Merge Real Vegas Lines (History)
    prop_hist_path = cfg.MASTER_PROP_HISTORY_FILE
    if prop_hist_path.exists():
        logging.info("Merging real historical FGA lines...")
        try:
            prop_hist = pd.read_parquet(prop_hist_path)
            
            if Cols.DATE in prop_hist.columns:
                prop_hist[Cols.DATE] = pd.to_datetime(prop_hist[Cols.DATE]).dt.normalize()
            
            # Only keep FGA
            prop_hist = prop_hist[prop_hist[Cols.PROP_TYPE] == 'FGA'].copy()
            
            if Cols.PROP_LINE in prop_hist.columns:
                prop_hist = prop_hist.drop_duplicates(subset=[Cols.PLAYER_NAME, Cols.DATE])
                prop_hist.rename(columns={Cols.PROP_LINE: 'Line_FGA'}, inplace=True)
                
                if Cols.PLAYER_NAME in box_scores.columns:
                    box_scores = pd.merge(
                        box_scores,
                        prop_hist[[Cols.PLAYER_NAME, Cols.DATE, 'Line_FGA']],
                        on=[Cols.PLAYER_NAME, Cols.DATE],
                        how='left'
                    )
                    # Assign the official prop line column expected by the model
                    box_scores[Cols.PROP_LINE] = box_scores['Line_FGA']
        except Exception as e:
            logging.warning(f"Failed to merge prop history: {e}")

    # 3. Generate FGA Rolling Features
    logging.info("Calculating FGA rolling standard deviations and averages...")
    
    # Use the highly specific FGA rolling function we built in generator.py
    training_df = generator.add_rolling_fga_history(box_scores.copy())

    # 4. Save Final Dataset
    logging.info(f"Saving training set with {training_df.shape[1]} columns...")
    training_df.to_parquet(cfg.MASTER_TRAINING_FILE, index=False)
    logging.info(f"Saved to {cfg.MASTER_TRAINING_FILE}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    create_training_dataset()