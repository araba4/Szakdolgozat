import sys

sys.dont_write_bytecode = True

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ONLAB = ROOT / "Önlab eredmények"
ONLAB_DATA = ONLAB / "data"

OUT = HERE / "out"
OUT_DATA = OUT / "data"
OUT_MODELS = OUT / "models"
OUT_RESULTS = OUT / "results"

if str(ONLAB) not in sys.path:
    sys.path.insert(0, str(ONLAB))


def guard_path(path) -> str:
    p = Path(path).resolve()
    if p == ONLAB.resolve() or ONLAB.resolve() in p.parents:
        raise SystemExit(f"Írás megtagadva (csak olvasható mappa): {p}")
    return str(p)
