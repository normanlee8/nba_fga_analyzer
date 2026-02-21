import pandas as pd
import requests
import time
import logging
import sys
import re
import io
import hashlib
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta
from bs4 import BeautifulSoup, Comment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols

# --- CONFIGURATION ---

def get_season_config():
    now = datetime.now()
    if now.month >= 10:
        current_start_year = now.year
    else:
        current_start_year = now.year - 1
        
    current_end_year = current_start_year + 1
    prev_start_year = current_start_year - 1
    prev_end_year = current_start_year

    return [
        {
            "season_str": f"{prev_start_year}-{str(prev_end_year)[-2:]}",
            "bball_ref_year": prev_end_year,
            "is_current": False,
            "tr_date_param": f"{prev_end_year}-07-01" 
        },
        {
            "season_str": f"{current_start_year}-{str(current_end_year)[-2:]}",
            "bball_ref_year": current_end_year,
            "is_current": True,
            "tr_date_param": None
        }
    ]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Connection': 'keep-alive',
}

TEAMRANKINGS_SLUG_MAP = {
    "Possessions per Game": "possessions-per-game",
    "Field Goals Attempted per Game": "field-goals-attempted-per-game",
    "Opponent Field Goals Attempted per Game": "opponent-field-goals-attempted-per-game",
    "Opponent Possessions per Game": "opponent-possessions-per-game",
    "Average Scoring Margin": "average-scoring-margin",
}

MASTER_FILE_MAP = {
    "NBA Player Per Game Averages.csv": ("https://www.basketball-reference.com/leagues/NBA_{YEAR}_per_game.html", "per_game_stats"),
    "NBA Player Advanced Stats.csv": ("https://www.basketball-reference.com/leagues/NBA_{YEAR}_advanced.html", "advanced"),
}

# --- UTILITIES ---

def create_robust_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session

def save_clean_parquet(df, filename_stem, output_dir):
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{filename_stem.replace('.csv', '')}.parquet"
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
        df.to_parquet(file_path, index=False)
        logging.info(f"Saved: {file_path.name}")
    except Exception as e:
        logging.error(f"Failed to save {filename_stem}: {e}")

def deduplicate_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols[cols == dup].index.values.tolist()] = [
            dup if i == 0 else f"{dup}_{i}" for i in range(sum(cols == dup))
        ]
    df.columns = cols
    return df

def generate_player_id(name):
    """
    Creates a universal Integer ID from a player's name so we don't need NBA API IDs.
    """
    clean = str(name).lower().replace('.', '').replace('-', ' ').replace("'", "").strip()
    for suffix in [' jr', ' sr', ' ii', ' iii', ' iv']:
        if clean.endswith(suffix):
            clean = clean[:-len(suffix)]
    return int(hashlib.md5(clean.encode()).hexdigest()[:8], 16)

def normalize_team(abbr):
    mapping = {'GS': 'GSW', 'NY': 'NYK', 'SA': 'SAS', 'UTAH': 'UTA', 'NO': 'NOP', 'WSH': 'WAS', 'PHO': 'PHX', 'NJ': 'BKN', 'CHA': 'CHA'}
    return mapping.get(abbr.upper(), abbr.upper())

# --- SCRAPING FUNCTIONS ---

def scrape_injuries(session, output_dir):
    logging.info("Scraping Daily Injuries (CBS Sports)...")
    url = "https://www.cbssports.com/nba/injuries/"
    try:
        response = session.get(url, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        all_rows = []
        for table in soup.find_all('table'):
            team_header = table.find_previous(class_="TeamLogoNameLockup-name")
            team_name = team_header.get_text(strip=True) if team_header else "UNK"
            
            for row in table.find_all('tr'):
                cols = row.find_all('td')
                if not cols: continue 
                
                name_cell = cols[0]
                long_name_span = name_cell.find('span', class_=lambda x: x and 'long' in x)
                player_text = long_name_span.get_text(strip=True) if long_name_span else name_cell.get_text(strip=True)

                all_rows.append({
                    "Team": team_name, "Player": player_text,
                    "Position": cols[1].get_text(strip=True),
                    "Status": cols[4].get_text(strip=True).upper()
                })

        if all_rows:
            df = pd.DataFrame(all_rows)
            df['Status_Clean'] = df['Status'].apply(lambda s: 'OUT' if 'OUT' in s else ('GTD' if 'QUESTIONABLE' in s or 'DECISION' in s else 'UNKNOWN'))
            save_clean_parquet(df, "daily_injuries", output_dir)
    except Exception as e:
        logging.error(f"Injury scrape failed: {e}")

def scrape_bball_ref(session, url_template, table_id, filename, year, output_dir):
    url = url_template.replace("{YEAR}", str(year))
    logging.info(f"Scraping Basketball-Reference {filename} for {year}...")
    
    try:
        response = session.get(url, timeout=20)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Bball-Ref hides large tables inside HTML comments. This unpacks them.
        table = soup.find('table', id=table_id)
        if not table:
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for c in comments:
                if f'id="{table_id}"' in c:
                    comment_soup = BeautifulSoup(c, 'html.parser')
                    table = comment_soup.find('table', id=table_id)
                    break
        
        if not table:
            logging.warning(f"Table {table_id} not found on {url}")
            return

        df = pd.read_html(io.StringIO(str(table)))[0]
        df = deduplicate_columns(df)
        df = df[df['Rk'] != 'Rk'].copy() 
        
        # Create Universal IDs for merging
        df['clean_name'] = df['Player'].apply(lambda x: str(x).lower().replace('.', '').replace('-', ' ').replace("'", "").strip())
        df['PLAYER_ID'] = df['clean_name'].apply(generate_player_id).astype('int64')
        
        save_clean_parquet(df, filename, output_dir)
            
    except Exception as e:
        logging.error(f"Bball-Ref scrape failed: {e}")

def scrape_teamrankings(session, slug, filename, date_param, output_dir):
    url = f"https://www.teamrankings.com/nba/stat/{slug}"
    if date_param: url += f"?date={date_param}"
    
    try:
        response = session.get(url, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table')
        if not table: return

        df = pd.read_html(io.StringIO(str(table)))[0]
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = [col[1] if len(col) > 1 else col[0] for col in df.columns]
        else:
            df.columns = [str(col) for col in df.columns]

        df = deduplicate_columns(df)
        if len(df.columns) >= 7:
            df = df.iloc[:, [0, 1, 2, 3, 4, 5, 6]].copy()
            df.columns = ["Rank", "Team", "Season", "Last 3", "Last 1", "Home", "Away"]
            df['Team'] = df['Team'].apply(lambda x: str(x).split('(')[0].strip())
            save_clean_parquet(df, filename, output_dir)
            
    except Exception as e:
        logging.error(f"TeamRankings {slug} failed: {e}")

def fetch_espn_season_box_scores(season_str, output_dir):
    """
    Mass-Fetches JSON Box Scores directly from ESPN's hidden public API.
    Bypasses NBA WAF, No Timeouts, extremely fast.
    """
    start_year = int(season_str.split('-')[0])
    start_date = datetime(start_year, 10, 15)
    end_date = datetime(start_year + 1, 6, 30)
    if end_date > datetime.now(): end_date = datetime.now()

    date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    session = create_robust_session()
    
    logging.info(f"Fetching ESPN schedule from {start_date.date()} to {end_date.date()}...")
    
    game_info_list = []
    def get_games(d):
        d_str = d.strftime("%Y%m%d")
        url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d_str}"
        try:
            data = session.get(url, timeout=10).json()
            if 'events' in data:
                return [(e['id'], d.strftime('%Y-%m-%d')) for e in data['events']]
        except:
            pass
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for games in executor.map(get_games, date_list):
            game_info_list.extend(games)

    logging.info(f"Found {len(game_info_list)} games. Fetching JSON Box Scores...")
    
    all_rows = []
    completed = 0
    total_games = len(game_info_list)

    def get_box(game_info):
        g_id, g_date = game_info
        url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={g_id}"
        try:
            data = session.get(url, timeout=10).json()
            rows = []
            if 'boxscore' not in data or 'players' not in data['boxscore']: return rows
            
            for team in data['boxscore']['players']:
                team_abbrev = normalize_team(team['team']['abbreviation'])
                
                if 'statistics' not in team or not team['statistics']: continue
                stats_keys = team['statistics'][0]['labels']
                
                try:
                    fg_idx = stats_keys.index('FG')
                    min_idx = stats_keys.index('MIN')
                except ValueError: continue
                    
                for athlete in team['statistics'][0]['athletes']:
                    stats = athlete['stats']
                    if not stats: continue
                    
                    fga = int(stats[fg_idx].split('-')[1]) if '-' in stats[fg_idx] else 0
                    try: minutes = float(stats[min_idx])
                    except: minutes = 0.0
                        
                    raw_name = athlete['athlete']['displayName']
                    p_id = generate_player_id(raw_name)
                    
                    rows.append({
                        'PLAYER_ID': p_id,
                        'PLAYER_NAME': raw_name,
                        'TEAM_ABBREVIATION': team_abbrev,
                        'GAME_ID': g_id,
                        Cols.DATE: g_date,
                        'MIN': minutes,
                        'FGA': fga
                    })
            return rows
        except:
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_game = {executor.submit(get_box, g): g for g in game_info_list}
        for future in concurrent.futures.as_completed(future_to_game):
            res = future.result()
            if res: all_rows.extend(res)
            
            completed += 1
            if completed % 100 == 0 or completed == total_games:
                logging.info(f"  -> Fetched {completed}/{total_games} ESPN box scores")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df['PLAYER_ID'] = df['PLAYER_ID'].astype('int64')
        df = df.sort_values(by=['PLAYER_ID', Cols.DATE]).reset_index(drop=True)
        save_clean_parquet(df, "NBA Player Box Scores", output_dir)
        logging.info(f"SUCCESS: Saved {len(df)} FGA Box Score records from ESPN.")
    else:
        logging.warning("No box scores fetched from ESPN.")

# --- MAIN EXECUTION ---

def main():
    start_time = time.time()
    common_setup()
    logging.info("========= STARTING ZERO-BLOAT API-FREE SCRAPER =========")
    
    session = create_robust_session()
    
    for season_cfg in get_season_config():
        season_str = season_cfg['season_str']
        is_current = season_cfg['is_current']
        
        output_dir = cfg.DATA_DIR / season_str
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"\n--- Processing Season: {season_str} ---")
        
        # 1. Injuries
        if is_current: scrape_injuries(session, output_dir)
        
        # 2. Basketball Reference (Usage Rates & Per Game)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            for filename, (url_temp, table_id) in MASTER_FILE_MAP.items():
                executor.submit(scrape_bball_ref, session, url_temp, table_id, filename, season_cfg['bball_ref_year'], output_dir)
                time.sleep(1)
        
        # 3. TeamRankings (Pace & Defense)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            for friendly_name, slug in TEAMRANKINGS_SLUG_MAP.items():
                sanitized_name = re.sub(r"\(.*\)", "", friendly_name).strip().replace(" / ", " per ").replace("/", " per ")
                filename = f"NBA Team {sanitized_name}" 
                executor.submit(scrape_teamrankings, session, slug, filename, season_cfg['tr_date_param'], output_dir)
                time.sleep(1) 
                
        # 4. ESPN Bulk Box Scores
        fetch_espn_season_box_scores(season_str, output_dir)
            
    session.close()
    
    elapsed = time.time() - start_time
    logging.info(f"\n========= FGA SCRAPER FINISHED IN {int(elapsed)} SECONDS =========")

def common_setup():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

if __name__ == "__main__":
    main()