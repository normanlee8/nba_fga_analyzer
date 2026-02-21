import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.data import etl, dataset
from prop_analyzer.utils import common

def main():
    common.setup_logging(name="build_fga_db")
    
    try:
        logging.info(">>> STARTING FGA DATABASE BUILD (Parquet Optimized) <<<")
        
        # --- PHASE 1: ETL (Extract, Transform, Load) ---
        season_folders = etl.get_season_folders(cfg.DATA_DIR)
        if not season_folders:
            logging.critical(f"No season folders found in {cfg.DATA_DIR}")
            return
            
        logging.info(f"Found Seasons: {[f.name for f in season_folders]}")

        # 1. Build Universal ID Map (Maps ESPN IDs to Bball-Ref)
        player_id_map = etl.create_player_id_map(season_folders)
        if player_id_map is None:
            logging.critical("Failed to create Player ID Map. Aborting.")
            return

        # 2. Process Master Stats
        etl.process_master_player_stats(player_id_map, season_folders, cfg.DATA_DIR)
        etl.process_master_team_stats(season_folders, cfg.DATA_DIR)
        etl.process_master_box_scores(season_folders, cfg.DATA_DIR)
        
        # 3. Process Defensive Context (Opponent FGA Allowed by Position)
        etl.process_dvp_stats(cfg.DATA_DIR)

        # --- PHASE 2: Dataset Creation ---
        logging.info("Step 2: Building Final FGA Training Dataset...")
        dataset.create_training_dataset()
        
        logging.info("<<< FGA DATABASE BUILD COMPLETE >>>")
        
    except Exception as e:
        logging.critical(f"FATAL ERROR in Database Build: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()