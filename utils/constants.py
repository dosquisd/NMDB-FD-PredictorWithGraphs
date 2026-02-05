from pathlib import Path

NAN_THRESHOLD: float = 0.5

ROOTDIR = Path(__file__).parents[1]
DATADIR = ROOTDIR / "data" / "ForbushDecrease"
MIN_VALUE_THRESHOLD: float = 1e-16
