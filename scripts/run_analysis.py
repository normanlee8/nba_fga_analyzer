import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.features import generator
from prop_analyzer.models import inference
from prop_analyzer.utils import common

def save_pretty_excel(df, output_path):
    """
    Saves the dataframe to Excel with Autosizing and a Color Scale on Confidence Score.
    """
    try:
        if df.empty: return

        has_xlsxwriter = False
        try:
            import xlsxwriter
            has_xlsxwriter = True
        except ImportError:
            logging.warning("XlsxWriter not installed. Saving standard CSV-style Excel.")

        if has_xlsxwriter:
            writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
            df.to_excel(writer, sheet_name='FGA_Picks', index=False)
            workbook = writer.book
            worksheet = writer.sheets['FGA_Picks']
            
            # Formats
            pct_fmt = workbook.add_format({'num_format': '0.0%'})
            float_fmt = workbook.add_format({'num_format': '0.00'})
            header_fmt = workbook.add_format({'bold': True, 'bottom': 1, 'bg_color': '#F0F0F0'})
            
            # Write Headers
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_fmt)

            # Auto-fit Columns
            for i, col in enumerate(df.columns):
                sample_values = df[col].astype(str).head(50)
                max_len = max(sample_values.map(len).max(), len(str(col)))
                width = min(max_len + 4, 40)
                
                if col == 'Model_Prob':
                    worksheet.set_column(i, i, width, pct_fmt)
                elif col in ['Expected_FGA', 'Edge', 'Volatility', 'Confidence_Score']:
                    worksheet.set_column(i, i, width, float_fmt)
                else:
                    worksheet.set_column(i, i, width)

            # Conditional Formatting: 3-Color Scale for Confidence Score
            conf_col_idx = df.columns.get_loc('Confidence_Score') if 'Confidence_Score' in df.columns else -1
            if conf_col_idx != -1:
                col_letter = xlsxwriter.utility.xl_col_to_name(conf_col_idx)
                rng = f"{col_letter}2:{col_letter}{len(df)+1}"
                
                worksheet.conditional_format(rng, {
                    'type': '3_color_scale',
                    'min_color': '#FFC7CE', # Red for low confidence
                    'mid_color': '#FFEB9C', # Yellow for medium
                    'max_color': '#C6EFCE'  # Green for high confidence
                })

            writer.close()
        else:
            df.to_excel(output_path, index=False)

        logging.info(f"Saved Excel analysis to: {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to save Excel file: {e}")

def print_pretty_table(df, title="TOP 15 FGA EDGES (BY CONFIDENCE SCORE)"):
    if df.empty:
        print("No results to display.")
        return

    df_str = df.astype(str)
    
    widths = []
    for col in df.columns:
        max_len = max(df_str[col].map(len).max(), len(col))
        widths.append(max_len + 2)

    fmt = "| " + " | ".join([f"{{:<{w}}}" for w in widths]) + " |"
    total_width = sum(widths) + (3 * len(widths)) + 1
    sep_line = "=" * total_width

    try:
        print(f"\n{title}")
        print(sep_line)
        print(fmt.format(*df.columns))
        print("-" * total_width)

        for _, row in df.iterrows():
            print(fmt.format(*row.values))

        print(sep_line + "\n")
    except Exception:
        print(df.head(15))

def main():
    common.setup_logging(name="analysis_pregame_fga")
    logging.info(">>> STARTING PRE-GAME FGA ANALYSIS <<<")
    
    try:
        # 1. Load Props
        props_path = cfg.PROPS_FILE
        if not props_path.exists():
            logging.critical(f"Props file not found: {props_path}")
            return

        try:
            props_df = pd.read_csv(props_path)
            props_df.columns = props_df.columns.str.strip()
            
            # Robust Date Parsing
            if Cols.DATE in props_df.columns:
                props_df[Cols.DATE] = pd.to_datetime(props_df[Cols.DATE], errors='coerce')
                if props_df[Cols.DATE].isna().any():
                    today = pd.Timestamp.now().normalize()
                    props_df[Cols.DATE] = props_df[Cols.DATE].fillna(today)
            else:
                props_df[Cols.DATE] = pd.Timestamp.now().normalize()
                
            logging.info(f"Loaded {len(props_df)} FGA props.")

            # History Loop
            try:
                history_path = cfg.MASTER_PROP_HISTORY_FILE
                history_entry = props_df.copy()
                
                if Cols.PLAYER_NAME in history_entry.columns:
                    history_entry[Cols.PLAYER_NAME] = history_entry[Cols.PLAYER_NAME].astype(str)
                if Cols.PROP_TYPE in history_entry.columns:
                    history_entry[Cols.PROP_TYPE] = history_entry[Cols.PROP_TYPE].astype(str)
                
                if history_path.exists():
                    existing_hist = pd.read_parquet(history_path)
                    combined_hist = pd.concat([existing_hist, history_entry], ignore_index=True)
                    
                    dedup_cols = [c for c in [Cols.PLAYER_NAME, Cols.DATE, Cols.PROP_TYPE] if c in combined_hist.columns]
                    if dedup_cols:
                        combined_hist.drop_duplicates(subset=dedup_cols, keep='last', inplace=True)
                    
                    combined_hist.to_parquet(history_path, index=False)
                else:
                    history_entry.to_parquet(history_path, index=False)
                    
            except Exception as e:
                logging.warning(f"Failed to save prop history: {e}")
            
        except Exception as e:
            logging.critical(f"Failed to read props file: {e}")
            return

        # 2. Build Features
        features_df = generator.build_feature_set(props_df)
        if features_df.empty: 
            logging.error("Feature generation returned empty dataframe.")
            return

        # 3. Run FGA Inference
        logging.info("Running FGA Machine Learning Inference...")
        results_df = inference.predict_props(features_df)
        
        if results_df is None or results_df.empty:
            logging.warning("No predictions were generated.")
            return

        # 4. Format Output
        if Cols.DATE in results_df.columns:
            results_df[Cols.DATE] = pd.to_datetime(results_df[Cols.DATE]).dt.strftime('%Y-%m-%d')

        rename_map = {
            Cols.PLAYER_NAME: 'Player',
            Cols.PROP_TYPE: 'Prop',
            Cols.PROP_LINE: 'Line',
            Cols.DATE: 'Date',
        }
        results_df.rename(columns=rename_map, inplace=True)

        keep_cols = [
            'Player', 'Team', 'Opponent', 'Line', 
            'Expected_FGA', 'Model_Prob', 'Pick', 
            'Edge', 'Volatility', 'Confidence_Score', 'Date'
        ]
        
        final_cols = [c for c in keep_cols if c in results_df.columns]
        final_output = results_df[final_cols].copy()

        # 5. Save Files
        final_output.to_parquet(cfg.PROCESSED_OUTPUT_SYSTEM, index=False)
        logging.info(f"Saved system results to {cfg.PROCESSED_OUTPUT_SYSTEM}")
        
        save_pretty_excel(final_output, cfg.PROCESSED_OUTPUT_XLSX)
        
        # 6. Console Display
        console_output = final_output.copy()
        if 'Model_Prob' in console_output.columns:
            if pd.api.types.is_numeric_dtype(console_output['Model_Prob']):
                console_output['Model_Prob'] = console_output['Model_Prob'].apply(lambda x: f"{x*100:.1f}%")
            
        print_pretty_table(console_output.head(15))

        logging.info("<<< FGA ANALYSIS COMPLETE >>>")
        
    except Exception as e:
        logging.critical(f"FATAL ERROR in Analysis Pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()