import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer.models import training
from prop_analyzer.utils import common

def main():
    # Setup logging
    common.setup_logging(name="run_training")
    
    try:
        # Route directly to the FGA optimized training module
        training.main()
        
    except Exception as e:
        logging.critical(f"FATAL ERROR in Training Pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()