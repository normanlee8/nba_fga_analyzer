# --- FEATURE GROUPS ---

# Features that should ALWAYS be included if available
BASE_FEATURE_COLS = [
    # Trend & Average
    'SZN Avg', 'L3 Avg', 'L5 EWMA', 'Loc Avg', 'CoV %',
    'SZN Games', 'Selected Std Dev', 'L10_STD_DEV',
    
    # Context
    'Prop Line',
    
    # Fatigue / Schedule
    'GAMES_IN_L5', 'IS_B2B', 'Days Rest',
    
    # Vacancy (The new split logic)
    'TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 
    'MISSING_USG_G', 'MISSING_USG_F',
    
    # Advanced
    'SZN_TS_PCT', 'SZN_EFG_PCT', 'SZN_USG_PROXY',
    'L5_TS_PCT', 'L5_EFG_PCT', 'L5_USG_PROXY',
    'LOC_TS_PCT', 'LOC_EFG_PCT', 'LOC_USG_PROXY',
    'TS_DIFF', 'EFG_DIFF', 'USG_DIFF',
    
    # PBP / OnCourt
    'PBP_OnCourt_PlusMinus', 'PBP_OnCourt_USG_PCT'
]

# Features that are specific to Matchup History (Prefix: VS_OPP_)
VS_OPP_FEATURES = [
    'VS_OPP_PTS', 'VS_OPP_REB', 'VS_OPP_AST', 'VS_OPP_STL', 
    'VS_OPP_BLK', 'VS_OPP_FG3M', 'VS_OPP_TOV', 
    'VS_OPP_PRA', 'VS_OPP_PR', 'VS_OPP_PA', 'VS_OPP_RA', 
    'VS_OPP_FANTASY_PTS', 'VS_OPP_MIN', 'VS_OPP_GAMES_PLAYED'
]

# Features regarding historical averages (Prefix: HIST_)
HIST_FEATURES = [
    'HIST_VS_OPP_PTS_AVG', 'HIST_VS_OPP_REB_AVG', 'HIST_VS_OPP_AST_AVG',
    'HIST_VS_OPP_PRA_AVG', 'HIST_VS_OPP_PR_AVG', 'HIST_VS_OPP_PA_AVG',
    'HIST_VS_OPP_RA_AVG', 'HIST_VS_OPP_FG3M_AVG', 'HIST_VS_OPP_GAMES'
]

# --- MAPPINGS ---

# For Feature Pruning: Defines which VS_OPP and HIST stats are relevant 
# for a specific target prop. (e.g. Don't use VS_OPP_AST to predict BLOCKS)
PROP_FEATURE_MAP = {
    # Full Game
    'PTS': ['PTS', 'FG3M', 'PRA', 'PR', 'PA', 'USG_PROXY', 'TS_PCT'],
    'REB': ['REB', 'PRA', 'PR', 'RA'],
    'AST': ['AST', 'PRA', 'PA', 'RA', 'TOV'],
    'PRA': ['PRA', 'PTS', 'REB', 'AST', 'PR', 'PA', 'RA'],
    'PR':  ['PR', 'PTS', 'REB', 'PRA'],
    'PA':  ['PA', 'PTS', 'AST', 'PRA'],
    'RA':  ['RA', 'REB', 'AST', 'PRA'],
    'FG3M': ['FG3M', 'PTS', 'USG_PROXY'],
    'STL': ['STL', 'STK', 'TOV'], 
    'BLK': ['BLK', 'STK', 'REB'], 
    'TOV': ['TOV', 'AST', 'USG_PROXY'],
    'FANTASY_PTS': ['FANTASY_PTS', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV'],
    'OREB': ['OREB', 'REB', 'RA', 'PR'],
    'FTM': ['FTM', 'PTS', 'FGA'],
    'DD': ['DD', 'PTS', 'REB', 'AST'],
    'TD': ['TD', 'PTS', 'REB', 'AST'],
    'STK': ['STK', 'STL', 'BLK'],
    
    # 1st Quarter (Explicitly look for Q1 stats)
    'Q1_PTS': ['Q1_PTS', 'Q1_FG3M', 'Q1_PRA', 'PTS'],
    'Q1_REB': ['Q1_REB', 'Q1_PRA', 'REB'],
    'Q1_AST': ['Q1_AST', 'Q1_PRA', 'AST'],
    'Q1_PRA': ['Q1_PRA', 'Q1_PTS', 'Q1_REB', 'Q1_AST', 'PRA'],
    'Q1_PR':  ['Q1_PR', 'Q1_PTS', 'Q1_REB', 'Q1_PRA', 'PR'],  # <--- Added
    'Q1_PA':  ['Q1_PA', 'Q1_PTS', 'Q1_AST', 'Q1_PRA', 'PA'],  # <--- Added
    'Q1_RA':  ['Q1_RA', 'Q1_REB', 'Q1_AST', 'Q1_PRA', 'RA'],  # <--- Added
    'Q1_FG3M': ['Q1_FG3M', 'Q1_PTS', 'FG3M'],
    
    # 1st Half
    '1H_PTS': ['1H_PTS', '1H_FG3M', '1H_PRA', 'PTS'],
    '1H_REB': ['1H_REB', '1H_PRA', 'REB'],
    '1H_AST': ['1H_AST', '1H_PRA', 'AST'],
    '1H_PRA': ['1H_PRA', '1H_PTS', '1H_REB', '1H_AST', 'PRA'],
    '1H_PR':  ['1H_PR', '1H_PTS', '1H_REB', '1H_PRA', 'PR'],  # <--- Added
    '1H_PA':  ['1H_PA', '1H_PTS', '1H_AST', '1H_PRA', 'PA'],  # <--- Added
    '1H_RA':  ['1H_RA', '1H_REB', '1H_AST', '1H_PRA', 'RA'],  # <--- Added
    '1H_FG3M': ['1H_FG3M', '1H_PTS', 'FG3M'],
}

# Keywords to look for in "Team Stats" columns (e.g. Rank columns)
# If we are training 'REB', we want columns containing 'Rebound' or 'REB'
RELEVANT_KEYWORDS = {
    'PTS': ['Points', 'PTS', 'Offensive Efficiency', 'True Shooting'],
    'REB': ['Rebound', 'REB', 'Opponent Total Rebounds'],
    'AST': ['Assist', 'AST', 'Turnover'],
    'FG3M': ['Three', '3P', 'FG3'],
    'TOV': ['Turnover', 'TOV', 'Steal'],
    'STL': ['Steal', 'Turnover', 'Opponent Turnovers'],
    'BLK': ['Block', 'BLK', 'Opponent Field Goals Attempted'],
    'PRA': ['Points', 'Rebound', 'Assist', 'Offensive Efficiency'],
    'PR': ['Points', 'Rebound'],
    'PA': ['Points', 'Assist'],
    'RA': ['Rebound', 'Assist'],
    'OREB': ['Offensive Rebound', 'OREB', 'Opponent Defensive Rebounds'],
    'FTM': ['Free Throw', 'FT', 'Opponent Personal Fouls'],
    'DD': ['Points', 'Rebound', 'Assist'],
    'TD': ['Points', 'Rebound', 'Assist'],
    'STK': ['Steal', 'Block'],
    
    # Q1 Specific Keywords
    'Q1_PTS': ['1st Quarter Points', '1Q', 'Points'],
    'Q1_REB': ['1st Quarter', '1Q', 'Rebounds'],
    'Q1_AST': ['1st Quarter', '1Q', 'Assists'],
    'Q1_PRA': ['1st Quarter', '1Q', 'Points', 'Rebounds', 'Assists'],
    'Q1_PR': ['1st Quarter', '1Q', 'Points', 'Rebounds'],
    'Q1_PA': ['1st Quarter', '1Q', 'Points', 'Assists'],
    'Q1_RA': ['1st Quarter', '1Q', 'Rebounds', 'Assists'],
    'Q1_FG3M': ['1st Quarter', '1Q', 'Three'],
    
    # 1H Specific Keywords
    '1H_PTS': ['1st Half Points', '1H', 'Points'],
    '1H_REB': ['1st Half', '1H', 'Rebounds'],
    '1H_AST': ['1st Half', '1H', 'Assists'],
    '1H_PRA': ['1st Half', '1H', 'Points', 'Rebounds', 'Assists'],
    '1H_PR': ['1st Half', '1H', 'Points', 'Rebounds'],
    '1H_PA': ['1st Half', '1H', 'Points', 'Assists'],
    '1H_RA': ['1st Half', '1H', 'Rebounds', 'Assists'],
    '1H_FG3M': ['1st Half', '1H', 'Three'],
}