import numpy as np
import pandas as pd
import logging

def calculate_slope(series):
    """
    Calculates the slope of the linear regression line for the series.
    Positive slope = Trending Up in FGA volume. Negative = Trending Down.
    """
    y = series.dropna().values
    n = len(y)
    if n < 2:
        return 0.0
    
    x = np.arange(n)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
    return slope

def calculate_blowout_risk_multiplier(team_margin, opp_margin, vegas_spread=None):
    """
    Penalizes expected minutes/FGA if the game is highly likely to be a blowout.
    High spread -> Starters sit the 4th quarter.
    """
    if vegas_spread is not None and not pd.isna(vegas_spread):
        projected_margin = abs(float(vegas_spread))
    else:
        projected_margin = abs(float(team_margin) - float(opp_margin))
        
    if projected_margin <= 10:
        return 1.0 # Competitive game, full minutes/FGA expectation
    elif projected_margin >= 20:
        return 0.75 # Massive blowout risk, 25% FGA haircut (missing most of 4th Q)
    else:
        # Linear scale between 10 and 20 points
        return 1.0 - ((projected_margin - 10) * 0.025)

def calculate_fga_metrics(history_df, stat_col='FGA'):
    """
    Specifically analyzes FGA standard deviation over 5, 10, and 20 game samples
    to feed the Confidence Score algorithm.
    """
    if history_df is None or history_df.empty or stat_col not in history_df.columns:
        return {'fga_avg': 0.0, 'fga_volatility': 1.0, 'recent_fga_trend': 0.0, 'fga_count': 0}

    data = history_df[stat_col].dropna()
    n = len(data)
    if n < 3:
        return {
            'fga_avg': data.mean() if n > 0 else 0.0, 
            'fga_volatility': 1.0, 
            'recent_fga_trend': 0.0, 
            'fga_count': n
        }

    # Calculate Volume Volatility at different time horizons
    std_5 = data.tail(5).std(ddof=1) if n >= 5 else data.std(ddof=1)
    std_10 = data.tail(10).std(ddof=1) if n >= 10 else data.std(ddof=1)
    std_20 = data.tail(20).std(ddof=1) if n >= 20 else data.std(ddof=1)
    
    # Bayesian Smoothing: Blended volatility heavily weighted towards recent form
    blended_volatility = (std_5 * 0.5) + (std_10 * 0.3) + (std_20 * 0.2)
    
    # Safety catch to prevent division by zero in confidence score math
    if pd.isna(blended_volatility) or blended_volatility <= 0:
        blended_volatility = 1.0

    return {
        'fga_avg': data.mean(),
        'fga_volatility': blended_volatility,
        'recent_fga_trend': calculate_slope(data.tail(10)),
        'fga_count': n
    }

def calculate_usage_vacancy_fga(team_roster_df):
    """
    Calculates the 'Vacancy' (Missing Usage) for a team based on current injuries.
    Aggregates by Position (Guard/Forward/Center) to contextually redistribute FGA.
    """
    metrics = {
        'TEAM_MISSING_USG': 0.0,
        'MISSING_USG_G': 0.0,
        'MISSING_USG_F': 0.0,
        'MISSING_USG_C': 0.0
    }
    
    if team_roster_df is None or team_roster_df.empty:
        return metrics
    
    if not all(col in team_roster_df.columns for col in ['STATUS', 'USG%', 'Pos']):
        return metrics

    # Normalize Status
    def get_injury_weight(status):
        s = str(status).upper().strip()
        if s in ['OUT', 'GTD']: return 1.0
        if 'DOUBTFUL' in s: return 0.75
        if 'QUESTIONABLE' in s: return 0.50
        return 0.0

    df = team_roster_df.copy()
    df['USG%'] = pd.to_numeric(df['USG%'], errors='coerce').fillna(0)
    df['Impact_Weight'] = df['STATUS'].apply(get_injury_weight)
    
    injured_df = df[df['Impact_Weight'] > 0].copy()
    if injured_df.empty:
        return metrics

    # Team Totals
    metrics['TEAM_MISSING_USG'] = (injured_df['USG%'] * injured_df['Impact_Weight']).sum()
    
    # Positional Breakdowns
    def categorize_pos(p):
        p = str(p).upper()
        if 'G' in p: return 'G'
        if 'F' in p: return 'F'
        if 'C' in p: return 'C'
        return 'X'

    injured_df['Gen_Pos'] = injured_df['Pos'].apply(categorize_pos)
    
    for pos_code in ['G', 'F', 'C']:
        pos_mask = injured_df['Gen_Pos'] == pos_code
        val = (injured_df.loc[pos_mask, 'USG%'] * injured_df.loc[pos_mask, 'Impact_Weight']).sum()
        metrics[f'MISSING_USG_{pos_code}'] = val

    return metrics

def calculate_expected_fga(base_fga, base_usg, team_missing_usg, pos_absorption_rate, team_pace, opp_pace, opp_def_fga_multiplier, blowout_multiplier):
    """
    The Core FGA Confidence Algorithm Math.
    Calculates Expected FGA based on Usage adjustments, Pace shifts, and Blowout risks.
    """
    # Defensive checks for missing data
    if base_usg <= 0: base_usg = 15.0 # League average proxy
    if team_pace <= 0: team_pace = 100.0
    if opp_pace <= 0: opp_pace = 100.0
    if pd.isna(opp_def_fga_multiplier): opp_def_fga_multiplier = 1.0

    # Step 1: Usage Redistribution
    usg_adj = base_usg + (team_missing_usg * pos_absorption_rate)
    usg_ratio = usg_adj / base_usg
    
    # Step 2: Pace Adjustment
    avg_game_pace = (team_pace + opp_pace) / 2.0
    pace_multiplier = avg_game_pace / team_pace
    
    # Step 3: Matchup & Script Combination
    fga_base_adjusted = base_fga * usg_ratio * pace_multiplier * opp_def_fga_multiplier
    
    # Step 4: Final Expected FGA considering minute limits
    expected_fga = fga_base_adjusted * blowout_multiplier
    
    return expected_fga