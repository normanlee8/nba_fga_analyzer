from pathlib import Path

# --- PATHS ---
BASE_DIR = Path(".")

# Data Directories
DATA_DIR = BASE_DIR / "prop_data"
MODEL_DIR = BASE_DIR / "prop_models"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
GRADED_DIR = OUTPUT_DIR / "graded_history"

# Ensure key directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(INPUT_DIR / "records").mkdir(parents=True, exist_ok=True)
GRADED_DIR.mkdir(parents=True, exist_ok=True)

# Specific File Paths
INPUT_PROPS_TXT = INPUT_DIR / "props_input.txt"
PROPS_FILE = INPUT_DIR / "props_today.csv"

# Final results
PROCESSED_OUTPUT_SYSTEM = OUTPUT_DIR / "processed_props.parquet" 
PROCESSED_OUTPUT_XLSX = OUTPUT_DIR / "processed_props.xlsx"

# --- MASTER DATA FILES (ALL PARQUET) ---
MASTER_PLAYER_FILE = DATA_DIR / "master_player_stats_2025-26.parquet"
MASTER_PLAYER_PATTERN = "master_player_stats_*.parquet"

MASTER_TEAM_FILE = DATA_DIR / "master_team_stats_2025-26.parquet"
MASTER_TEAM_PATTERN = "master_team_stats_*.parquet"

MASTER_BOX_SCORES_FILE = DATA_DIR / "master_box_scores_2025-26.parquet"
MASTER_BOX_SCORES_PATTERN = "master_box_scores_*.parquet"

# Master Prop History
MASTER_PROP_HISTORY_FILE = DATA_DIR / "master_prop_history.parquet"

# Opponent & Matchup Context
MASTER_VS_OPP_FILE = DATA_DIR / "master_vs_opponent.parquet"
MASTER_DVP_FILE = DATA_DIR / "master_dvp_stats.parquet"
MASTER_TRAINING_FILE = DATA_DIR / "master_training_dataset.parquet"

# --- DATA CONTRACT (SCHEMA) ---
class Cols:
    PLAYER_NAME = 'Player Name'
    PLAYER_ID = 'PLAYER_ID'
    GAME_ID = 'GAME_ID' 
    TEAM = 'Team'
    OPPONENT = 'Opponent'
    MATCHUP = 'Matchup'
    DATE = 'GAME_DATE'
    
    PROP_TYPE = 'Prop Category'
    PROP_LINE = 'Prop Line'
    
    PREDICTION = 'Expected_FGA'
    CONFIDENCE = 'Confidence_Score'
    
    ACTUAL_VAL = 'Actual Value'
    RESULT = 'Result'
    CORRECTNESS = 'Correctness'
    
    @classmethod
    def get_required_input_cols(cls):
        return [cls.PLAYER_NAME, cls.TEAM, cls.OPPONENT, cls.MATCHUP, cls.PROP_TYPE, cls.PROP_LINE, cls.DATE]

# --- TUNING / CONSTANTS ---
LIVE_BLOWOUT_THRESHOLD = 20
MIN_GAMES_FOR_ANALYSIS = 5

# Hardcoded FGA support to prevent errors in generic scripts
MASTER_PROP_MAP = {
    'FG Attempted': 'FGA', 'Field Goals Attempted': 'FGA', 'FGA': 'FGA'
}

SUPPORTED_PROPS = ['FGA']