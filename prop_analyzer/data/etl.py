import pandas as pd
import logging
import re
from pathlib import Path
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols

def get_season_folders(data_dir):
    folders = [f for f in data_dir.iterdir() if f.is_dir() and re.match(r'\d{4}-\d{2}', f.name)]
    return sorted(folders)

def create_player_id_map(season_folders):
    """Pulls universal IDs created by our ESPN scraper."""
    logging.info("Building Player ID Map from ESPN Box Scores...")
    dfs = []
    for folder in season_folders:
        bs_path = folder / "NBA Player Box Scores.parquet"
        if bs_path.exists():
            df = pd.read_parquet(bs_path)
            if Cols.PLAYER_ID in df.columns:
                dfs.append(df[[Cols.PLAYER_ID, 'PLAYER_NAME', 'TEAM_ABBREVIATION']].drop_duplicates(subset=[Cols.PLAYER_ID]))
    
    if not dfs: return None
    id_map = pd.concat(dfs).drop_duplicates(subset=[Cols.PLAYER_ID], keep='last')
    id_map['clean_name'] = id_map['PLAYER_NAME'].apply(lambda x: str(x).lower().replace('.', '').replace('-', ' ').replace("'", "").strip())
    return id_map

def process_master_player_stats(player_id_map, season_folders, output_dir):
    """Merges Basketball-Reference Usage and Per Game stats to ESPN IDs."""
    logging.info("Processing Master Player Stats (Usage Rates)...")
    for folder in season_folders:
        season_id = folder.name
        try:
            pg_path = folder / "NBA Player Per Game Averages.parquet"
            adv_path = folder / "NBA Player Advanced Stats.parquet"
            
            if not pg_path.exists() or not adv_path.exists(): continue
            
            pg_df = pd.read_parquet(pg_path)
            adv_df = pd.read_parquet(adv_path)
            
            # Keep Name mappings so generator.py can translate Underdog text props to IDs
            if 'Player' in pg_df.columns:
                pg_df['PLAYER_NAME'] = pg_df['Player']
                
            pg_cols = [c for c in [Cols.PLAYER_ID, 'PLAYER_NAME', 'clean_name', 'Pos'] if c in pg_df.columns]
            pg_subset = pg_df[pg_cols].copy() if Cols.PLAYER_ID in pg_df.columns else pd.DataFrame()
            
            adv_subset = adv_df[[Cols.PLAYER_ID, 'USG%']].copy() if Cols.PLAYER_ID in adv_df.columns else pd.DataFrame()
            
            if pg_subset.empty or adv_subset.empty: continue
            
            merged = pd.merge(pg_subset, adv_subset, on=Cols.PLAYER_ID, how='inner').drop_duplicates(subset=[Cols.PLAYER_ID])
            merged['SEASON_ID'] = season_id
            
            merged.to_parquet(output_dir / f"master_player_stats_{season_id}.parquet", index=False)
        except Exception as e:
            logging.error(f"Error processing player stats for {season_id}: {e}")

def process_master_team_stats(season_folders, output_dir):
    """Merges TeamRankings Pace and Defense."""
    logging.info("Processing Master Team Stats (Pace & Defense)...")
    for folder in season_folders:
        season_id = folder.name
        team_dfs = []
        for f in folder.glob("NBA Team*.parquet"):
            try:
                df = pd.read_parquet(f)
                metric_name = f.stem.replace("NBA Team ", "")
                val_col = [c for c in df.columns if re.match(r'202\d', str(c))][0]
                df[metric_name] = pd.to_numeric(df[val_col], errors='coerce')
                team_dfs.append(df[['Team', metric_name]].rename(columns={'Team': 'TEAM_ABBREVIATION'}))
            except: pass
            
        if team_dfs:
            master = team_dfs[0]
            for df in team_dfs[1:]:
                master = pd.merge(master, df, on='TEAM_ABBREVIATION', how='outer')
            master['SEASON_ID'] = season_id
            master.to_parquet(output_dir / f"master_team_stats_{season_id}.parquet", index=False)

def process_master_box_scores(season_folders, output_dir):
    """Cleans ESPN Box Scores."""
    logging.info("Processing Master Box Scores...")
    for folder in season_folders:
        season_id = folder.name
        bs_path = folder / "NBA Player Box Scores.parquet"
        if not bs_path.exists(): continue
        
        try:
            df = pd.read_parquet(bs_path)
            df['SEASON_ID'] = season_id
            df[Cols.DATE] = pd.to_datetime(df[Cols.DATE]).dt.strftime('%Y-%m-%d')
            
            # Bring in Positions
            p_stats_path = output_dir / f"master_player_stats_{season_id}.parquet"
            if p_stats_path.exists():
                p_stats = pd.read_parquet(p_stats_path)
                df = pd.merge(df, p_stats[[Cols.PLAYER_ID, 'Pos']], on=Cols.PLAYER_ID, how='left')
                df['Pos'] = df['Pos'].fillna('PG')
            
            # Simple ESPN Matchup Parsing
            df.to_parquet(output_dir / f"master_box_scores_{season_id}.parquet", index=False)
        except Exception as e:
            logging.error(f"Box score processing failed for {season_id}: {e}")

def process_dvp_stats(output_dir):
    """Calculates Opponent FGA Allowed by Position."""
    logging.info("Processing Defense vs Position (DvP) for FGA...")
    files = sorted(output_dir.glob("master_box_scores_*.parquet"))
    all_dvp = []
    
    for f in files:
        try:
            season_id = re.search(r'\d{4}-\d{2}', f.name).group(0)
            df = pd.read_parquet(f)
            
            if 'FGA' not in df.columns or 'Pos' not in df.columns: continue
            
            # Normalize Position Bucket
            df['Primary_Pos'] = df['Pos'].apply(lambda x: str(x).split('-')[0].upper().strip() if isinstance(x, str) else 'PG')
            df['Primary_Pos'] = df['Primary_Pos'].replace({'G': 'SG', 'F': 'PF'})
            
            # Opponent is the team in the game that isn't the player's team
            # Simplified for bulk stats: Group FGA allowed by Team Abbreviation playing defense
            # Note: Because we have raw ESPN boxes without direct opponent maps in this simplified view,
            # we will rely on TeamRankings Opponent FGA mapping in feature generation instead.
            pass
        except: pass