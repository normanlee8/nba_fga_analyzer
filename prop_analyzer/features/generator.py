import pandas as pd
import numpy as np
import logging
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.data import loader

def add_rolling_fga_history(df):
    """
    Calculates historical rolling features specifically for Field Goal Attempts (FGA).
    CRITICAL: All rolling stats are shifted by 1 to represent "stats entering the game"
    to prevent future-data leakage during backtesting and training.
    """
    if Cols.PLAYER_ID not in df.columns or Cols.DATE not in df.columns:
        logging.error(f"Missing ID/Date columns. Cols found: {df.columns}")
        return df

    # Strict Multi-Level Sort to prevent leakage
    sort_cols = [Cols.PLAYER_ID, Cols.DATE]
    if Cols.GAME_ID in df.columns:
        sort_cols.append(Cols.GAME_ID)
        
    df = df.sort_values(by=sort_cols).reset_index(drop=True)
    
    # Ensure FGA exists
    if 'FGA' not in df.columns: 
        df['FGA'] = 0.0

    grouped = df.groupby(Cols.PLAYER_ID)['FGA']

    # FGA Moving Averages
    df['FGA_SZN_AVG'] = grouped.expanding().mean().shift(1).values
    df['FGA_L5_AVG'] = grouped.rolling(window=5, min_periods=1).mean().shift(1).values
    df['FGA_L10_AVG'] = grouped.rolling(window=10, min_periods=1).mean().shift(1).values
    
    # FGA Volatility (Standard Deviation) - Feeds the Confidence Algorithm
    df['FGA_L5_STD'] = grouped.rolling(window=5, min_periods=2).std().shift(1).values
    df['FGA_L10_STD'] = grouped.rolling(window=10, min_periods=3).std().shift(1).values
    df['FGA_L20_STD'] = grouped.rolling(window=20, min_periods=5).std().shift(1).values

    # Advanced Stats (Usage Rate Proxy)
    if 'USG_PROXY' in df.columns:
        df['SZN_USG_PROXY'] = df.groupby(Cols.PLAYER_ID)['USG_PROXY'].expanding().mean().shift(1).values
        df['L5_USG_PROXY'] = df.groupby(Cols.PLAYER_ID)['USG_PROXY'].rolling(window=5).mean().shift(1).values
        
    return df

def build_feature_set(props_df):
    logging.info("Building FGA-specific feature set with Point-in-Time safety...")
    
    # 1. Load Data
    player_stats_static, team_stats, _ = loader.load_static_data()
    vs_opp_df = loader.load_vs_opponent_data()
    
    if props_df.empty:
        return pd.DataFrame()

    box_scores = loader.load_box_scores()
    
    dvp_df = None
    if cfg.MASTER_DVP_FILE.exists():
        try:
            dvp_df = pd.read_parquet(cfg.MASTER_DVP_FILE)
        except Exception as e:
            logging.error(f"Failed to read DVP Parquet: {e}")

    # 2. Map Player Names to IDs
    if Cols.PLAYER_ID not in props_df.columns:
        if player_stats_static is not None:
            name_map = player_stats_static.set_index('clean_name')[Cols.PLAYER_ID].to_dict()
            props_df['clean_name'] = props_df[Cols.PLAYER_NAME].apply(lambda x: str(x).lower().strip())
            
            # Manual Mapping overrides
            manual_map = {
                'deuce mcbride': 'miles mcbride',
                'cam johnson': 'cameron johnson',
                'lu dort': 'luguentz dort',
                'pj washington': 'p.j. washington',
                'jimmy butler': 'jimmy butler iii',
                'herb jones': 'herbert jones',
                'robert williams': 'robert williams iii',
                'trey murphy': 'trey murphy iii',
                'kelly oubre': 'kelly oubre jr.',
                'michael porter': 'michael porter jr.',
                'gg jackson': 'gg jackson ii'
            }
            props_df['clean_name'] = props_df['clean_name'].replace(manual_map)
            props_df[Cols.PLAYER_ID] = props_df['clean_name'].map(name_map)
            
            props_df = props_df.dropna(subset=[Cols.PLAYER_ID]).copy()
            if props_df.empty: 
                logging.warning("No players matched ID map. Check naming conventions.")
                return pd.DataFrame()

            props_df[Cols.PLAYER_ID] = props_df[Cols.PLAYER_ID].astype('int64')
        else:
            logging.error("Static player stats missing. Cannot map IDs.")
            return pd.DataFrame()

    # 3. Time-Travel Feature Engineering (Full Game FGA)
    if box_scores is not None and not box_scores.empty:
        logging.info("Calculating Full Game FGA rolling stats...")
        
        box_scores[Cols.PLAYER_ID] = box_scores[Cols.PLAYER_ID].fillna(0).astype('int64')
        props_df[Cols.PLAYER_ID] = props_df[Cols.PLAYER_ID].astype('int64')
        
        if Cols.DATE in box_scores.columns:
            box_scores[Cols.DATE] = pd.to_datetime(box_scores[Cols.DATE])
        elif 'GAME_DATE' in box_scores.columns:
             box_scores[Cols.DATE] = pd.to_datetime(box_scores['GAME_DATE'])

        # Calculate history with shifts
        history_df = add_rolling_fga_history(box_scores.copy())
        
        props_df[Cols.DATE] = pd.to_datetime(props_df[Cols.DATE])
        history_df[Cols.DATE] = pd.to_datetime(history_df[Cols.DATE])
        
        props_df = props_df.sort_values(Cols.DATE)
        history_df = history_df.sort_values(Cols.DATE)
        
        # Merge point-in-time stats
        features_df = pd.merge_asof(
            props_df, history_df, on=Cols.DATE, by=Cols.PLAYER_ID,
            direction='backward', suffixes=('', '_hist')
        )
        
        # Merge Static Stats (Current Season USG% proxy)
        if player_stats_static is not None:
            cols_to_use = [c for c in player_stats_static.columns 
                           if c not in features_df.columns or c == Cols.PLAYER_ID]
            features_df = pd.merge(features_df, player_stats_static[cols_to_use], on=Cols.PLAYER_ID, how='left')
    else:
        features_df = pd.merge(props_df, player_stats_static, on=Cols.PLAYER_ID, how='left')

    # 4. Merge Team/Opponent Stats (Pace & Matchup Context)
    if 'TEAM_ABBREVIATION' not in features_df.columns and Cols.TEAM in features_df.columns:
        features_df['TEAM_ABBREVIATION'] = features_df[Cols.TEAM]
        
    if team_stats is not None:
        team_stats_renamed = team_stats.add_prefix('TEAM_')
        if 'TEAM_TEAM_ABBREVIATION' in team_stats_renamed.columns:
             team_stats_renamed = team_stats_renamed.rename(columns={'TEAM_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION'})
        
        # Merge Team Stats (Pace, Margin)
        features_df = pd.merge(features_df, team_stats_renamed, left_on='TEAM_ABBREVIATION', right_index=True, how='left')
        
        # Merge Opponent Stats (Opp Pace, Opp FGA Allowed)
        opp_stats_renamed = team_stats.add_prefix('OPP_')
        features_df = pd.merge(features_df, opp_stats_renamed, left_on=Cols.OPPONENT, right_index=True, how='left')

    # 5. Merge DVP (Defensive FGA Allowed by Position)
    if dvp_df is not None:
        if 'Pos' not in features_df.columns and player_stats_static is not None:
             if Cols.PLAYER_ID in player_stats_static.columns:
                 pos_map = player_stats_static.set_index(Cols.PLAYER_ID)['Pos'].to_dict()
                 features_df['Pos'] = features_df[Cols.PLAYER_ID].map(pos_map).fillna('PG')

        def normalize_pos(p):
            p = str(p).split('-')[0].upper().strip()
            if p == 'G': return 'SG'
            if p == 'F': return 'PF'
            return p if p in ['PG','SG','SF','PF','C'] else 'PG'
            
        features_df['Primary_Pos'] = features_df.get('Pos', 'PG').apply(normalize_pos)
        features_df['Primary_Pos'] = features_df['Primary_Pos'].astype(str)
        
        if 'Primary_Pos' in dvp_df.columns:
            dvp_df['Primary_Pos'] = dvp_df['Primary_Pos'].astype(str)
        
        if 'SEASON_ID' not in features_df.columns:
             features_df['yr'] = features_df[Cols.DATE].dt.year
             features_df['mo'] = features_df[Cols.DATE].dt.month
             features_df['season_start'] = np.where(features_df['mo'] > 8, features_df['yr'], features_df['yr'] - 1)
             features_df['SEASON_ID'] = features_df['season_start'].astype(str) + "-" + (features_df['season_start'] + 1).astype(str).str[-2:]
             features_df.drop(columns=['yr', 'mo', 'season_start'], inplace=True)

        if 'SEASON_ID' in dvp_df.columns:
            features_df = pd.merge(
                features_df, dvp_df, 
                left_on=['SEASON_ID', Cols.OPPONENT, 'Primary_Pos'], 
                right_on=['SEASON_ID', 'OPPONENT_ABBREV', 'Primary_Pos'], 
                how='left'
            )
        else:
            features_df = pd.merge(
                features_df, dvp_df, 
                left_on=[Cols.OPPONENT, 'Primary_Pos'], 
                right_on=['OPPONENT_ABBREV', 'Primary_Pos'], 
                how='left'
            )

    # 6. Merge H2H (Head to Head FGA History)
    if vs_opp_df is not None and not vs_opp_df.empty:
        features_df = pd.merge(
            features_df, vs_opp_df,
            left_on=[Cols.PLAYER_ID, Cols.OPPONENT],
            right_on=[Cols.PLAYER_ID, 'OPPONENT_ABBREV'],
            how='left'
        )

    # 7. Final Polish / Fill Vacancy Default
    # Standardize Pace columns explicitly for the model
    if 'TEAM_Possessions per Game' in features_df.columns:
        features_df['TEAM_PACE'] = features_df['TEAM_Possessions per Game']
    if 'OPP_Possessions per Game' in features_df.columns:
        features_df['OPP_PACE'] = features_df['OPP_Possessions per Game']
        
    cols_to_fill = ['TEAM_MISSING_USG', 'MISSING_USG_G', 'MISSING_USG_F', 'MISSING_USG_C']
    for c in cols_to_fill:
        if c not in features_df.columns: features_df[c] = 0.0
        features_df[c] = features_df[c].fillna(0.0)

    # Ensure FGA Variance columns don't have NaNs
    variance_cols = ['FGA_L5_STD', 'FGA_L10_STD', 'FGA_L20_STD']
    for c in variance_cols:
        if c in features_df.columns:
            features_df[c] = features_df[c].fillna(1.0) # Baseline volatility 

    logging.info(f"FGA Feature set built. Final Shape: {features_df.shape}")
    return features_df