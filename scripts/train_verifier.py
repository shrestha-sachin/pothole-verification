"""CLI script to train MobileNetV3-Small pothole verifier.

Usage:
    python scripts/train_verifier.py --dry-run
    python scripts/train_verifier.py --output models/verifier_v1.pt
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.verifier.train import main

if __name__ == "__main__":
    main()

