import pandas as pd
import logging
import re
from pathlib import Path
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.utils.text import preprocess_name_for_fuzzy_match

_INJURY_CACHE = None
_INJURY_WARNING_SHOWN = False

def load_static_data():
    """Loads master player and team stats."""
    logging.info("--- Loading Static Data Files (Parquet) ---")
    try:
        # Player Stats
        player_files = sorted(cfg.DATA_DIR.glob(cfg.MASTER_PLAYER_PATTERN))
        player_dfs = [pd.read_parquet(f) for f in player_files] if player_files else []
        player_stats_df = pd.concat(player_dfs, ignore_index=True) if player_dfs else pd.DataFrame()
        
        if Cols.PLAYER_ID in player_stats_df.columns:
            player_stats_df = player_stats_df.drop_duplicates(subset=[Cols.PLAYER_ID], keep='last')
            
        # Team Stats
        team_files = sorted(cfg.DATA_DIR.glob(cfg.MASTER_TEAM_PATTERN))
        team_dfs = [pd.read_parquet(f) for f in team_files] if team_files else []
        team_stats_df = pd.concat(team_dfs, ignore_index=True) if team_dfs else pd.DataFrame()
        
        if 'TEAM_ABBREVIATION' in team_stats_df.columns:
            team_stats_df = team_stats_df.drop_duplicates(subset=['TEAM_ABBREVIATION'], keep='last')
            team_stats_df.set_index('TEAM_ABBREVIATION', inplace=True)
            
        return player_stats_df, team_stats_df, 100.0
    except Exception as e:
        logging.critical(f"FATAL: Failed to load static master files: {e}")
        return None, None, 100.0

def load_box_scores(player_ids=None):
    """Loads ALL master_box_scores_*.parquet files."""
    try:
        files = sorted(cfg.DATA_DIR.glob(cfg.MASTER_BOX_SCORES_PATTERN))
        if not files: return None

        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            if player_ids is not None and Cols.PLAYER_ID in df.columns:
                df = df[df[Cols.PLAYER_ID].isin(set(player_ids))]
            dfs.append(df)

        box_scores_df = pd.concat(dfs, ignore_index=True)

        if Cols.DATE in box_scores_df.columns:
            box_scores_df[Cols.DATE] = pd.to_datetime(box_scores_df[Cols.DATE], errors='coerce').dt.normalize()
            box_scores_df.dropna(subset=[Cols.DATE], inplace=True)
            box_scores_df.sort_values(by=Cols.DATE, ascending=False, inplace=True)
        
        return box_scores_df
    except Exception as e:
        logging.critical(f"FATAL: Failed to load box scores: {e}")
        return None

def load_vs_opponent_data():
    """Returns empty DF as we handle FGA Matchups dynamically now."""
    return pd.DataFrame()

def get_cached_injury_data():
    """Loads injury data from Parquet."""
    global _INJURY_CACHE, _INJURY_WARNING_SHOWN
    if _INJURY_CACHE is not None: return _INJURY_CACHE
    
    search_paths = []
    if cfg.DATA_DIR.exists():
        season_folders = sorted([f for f in cfg.DATA_DIR.iterdir() if f.is_dir() and re.match(r'\d{4}-\d{2}', f.name)], reverse=True)
        if season_folders:
            search_paths.append(season_folders[0] / "daily_injuries.parquet")
    search_paths.append(cfg.DATA_DIR / "daily_injuries.parquet")
    
    for p in search_paths:
        if p.exists():
            try:
                df = pd.read_parquet(p)
                if 'Status_Clean' not in df.columns and 'Injury Status' in df.columns:
                    df['Status_Clean'] = df['Injury Status'].apply(lambda x: 'OUT' if 'out' in str(x).lower() else 'GTD' if 'question' in str(x).lower() else 'UNKNOWN')
                _INJURY_CACHE = df
                return df
            except Exception as e:
                logging.warning(f"Failed to read {p}: {e}")
                
    if not _INJURY_WARNING_SHOWN:
        logging.warning("daily_injuries file not found.")
        _INJURY_WARNING_SHOWN = True
    return None