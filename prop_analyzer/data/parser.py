import csv
import re
import datetime
import logging
import pandas as pd
from pathlib import Path
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols

DAYS_MAP = {
    'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3,
    'Fri': 4, 'Sat': 5, 'Sun': 6,
    'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3,
    'FRI': 4, 'SAT': 5, 'SUN': 6
}

class SmartDateDetector:
    def __init__(self):
        pass

    def get_date_from_day(self, day_str):
        if not day_str or day_str not in DAYS_MAP:
            return None
            
        target_weekday = DAYS_MAP[day_str]
        today = datetime.datetime.now()
        current_weekday = today.weekday()
        
        diff = target_weekday - current_weekday
        if diff < -2: diff += 7
        elif diff > 4: diff -= 7
             
        target_date = today + datetime.timedelta(days=diff)
        return target_date.strftime("%Y-%m-%d")

    def find_date(self, team, opponent, day_str=None):
        if day_str:
            calculated_date = self.get_date_from_day(day_str)
            if calculated_date: return calculated_date
        return datetime.datetime.now().strftime("%Y-%m-%d")

def clean_prop_line(text):
    s = text.strip().upper()
    s = s.replace(',', '.') 
    s = re.sub(r'^[OU]\s+', '', s)
    s = s.replace('OVER', '').replace('UNDER', '').strip()
    try:
        val = float(s)
        return str(val)
    except ValueError:
        return None

def parse_matchup(matchup_line):
    day_match = re.search(r'-\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)', matchup_line, re.IGNORECASE)
    day_str = day_match.group(1).capitalize() if day_match else None

    line = matchup_line.replace(' vs ', ' @ ').replace(' vs. ', ' @ ').replace('-', ' @ ')
    match = re.search(r'\b([A-Z]{2,3})\s*@\s*([A-Z]{2,3})\b', line) 
    
    if match:
        team1 = match.group(1)
        team2 = match.group(2)
        return team1, team2, f"{team1} vs. {team2}", day_str
    return None, None, None, None

def parse_text_to_csv(input_path=None, output_path=None):
    if input_path is None: input_path = cfg.INPUT_PROPS_TXT
    if output_path is None: output_path = cfg.PROPS_FILE
    
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        return

    date_detector = SmartDateDetector()
    
    current_player, current_team, current_opponent = None, None, None
    current_matchup, current_game_date, prop_line_value = None, None, None
    
    data_to_write = [] 
    lines_processed, props_parsed = 0, 0

    IGNORED_PHRASES = {
        'HIGHER', 'LOWER', 'FEWER PICKS', 'MORE PICKS', 'DRAFTS', 
        'PICK\'EM', 'LIVE', 'RESULTS', 'RANKINGS', 'NEWS FEED', 
        '$0.00', 'ALL NBA', 'COLLAPSE ALL', 'ADD PICKS', 'ENTRY AMOUNT',
        'REWARDS', 'ENTER AMOUNT', 'STANDARD', 'FLEX', 'PLAY',
        'FIND NBA TEAMS', 'PRE-GAME & IN-GAME', 'PICK\'EM TIPS',
        'FIELD GOALS ATTEMPTED', 'FGA', 'FG ATTEMPTS' 
    }

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()

        # Check the first line for a hardcoded date override (e.g. "2026-02-20")
        manual_date = None
        if lines:
            first_line = lines[0].strip()
            date_match = re.search(r'(202\d-\d{2}-\d{2})', first_line)
            if date_match:
                manual_date = date_match.group(1)
                logging.info(f"Manual Date Override Detected! Forcing all props to: {manual_date}")

        for line in lines:
            line = line.strip()
            if not line: continue
            lines_processed += 1

            t1, t2, full_matchup, day_str = parse_matchup(line)
            if full_matchup:
                current_matchup = full_matchup
                current_team, current_opponent = t1, t2
                current_game_date = manual_date if manual_date else date_detector.find_date(t1, t2, day_str)
                continue 

            cleaned_val = clean_prop_line(line)
            if cleaned_val:
                prop_line_value = cleaned_val
                continue

            if prop_line_value is not None:
                if current_player and current_matchup:
                    data_to_write.append([current_player, current_team, current_opponent, current_matchup, 'FGA', prop_line_value, current_game_date])
                    props_parsed += 1
                prop_line_value = None 
                continue

            upper_line = line.upper()
            if upper_line in IGNORED_PHRASES or any(upper_line.startswith(p) for p in ['MORE PICKS', 'GET UP TO', 'CLAIM YOUR']): continue
            if any(char.isdigit() for char in line) or 'OVER' in upper_line or 'UNDER' in upper_line: continue
            
            current_player = line
            prop_line_value = None 

        if prop_line_value is not None and current_player and current_matchup:
            data_to_write.append([current_player, current_team, current_opponent, current_matchup, 'FGA', prop_line_value, current_game_date])
            props_parsed += 1

        if not data_to_write:
            logging.warning("No valid props parsed.")
            return

        header = Cols.get_required_input_cols()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(header)
            writer.writerows(data_to_write)
            
        logging.info(f"Successfully converted FGA props to {output_path} ({len(data_to_write)} rows)")

    except Exception as e:
        logging.error(f"Error parsing props: {e}", exc_info=True)