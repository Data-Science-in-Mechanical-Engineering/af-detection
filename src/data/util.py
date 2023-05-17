from __future__ import annotations

from pathlib import Path

if "__file__" in globals():
    DATA_PATH = Path(__file__).parent.parent.parent / "data"
else:
    DATA_PATH = Path().resolve().parent.parent.parent / "data"


class SPHPath:
    ROOT = DATA_PATH / "sph"
    RAW_DATA = ROOT / "raw"
    TRAIN_DATA = ROOT / "train"
    VALIDATE_DATA = ROOT / "validate"
    TEST_DATA = ROOT / "test"
    DENOISED_FOLDER = "ECGDataDenoised"
    NOISY_FOLDER = "ECGData"


class COATPath:
    ROOT = DATA_PATH / "coat"
    RAW_DATA = ROOT / "raw"
    TRAIN_DATA = ROOT / "train"
    VALIDATE_DATA = ROOT / "validate"
    TEST_DATA = ROOT / "test"


RANDOM_SEED = 42
