import re, os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, List

import pandas as pd


PARQUET_ENGINE: str = "pyarrow"
DATE_CANDIDATES: Sequence[str] = (
    "date", "data", "dia", "dt", "datetime", "timestamp", "ts"
)


def sanitize(s: str) -> str:
    """Normalize a string to safe snake_case-ish tokens."""
    return re.sub(r"[^\w]+", "_", str(s), flags=re.UNICODE).strip("_").lower()

def _check_engine(engine: str) -> None:
    if engine == "pyarrow":
        try:
            import pyarrow
        except Exception as e:
            raise RuntimeError("pyarrow not installed. Run: pip install pyarrow") from e
    elif engine == "fastparquet":
        try:
            import fastparquet
        except Exception as e:
            raise RuntimeError("fastparquet not installed. Run: pip install fastparquet") from e
    else:
        raise ValueError("engine must be 'pyarrow' or 'fastparquet'")

_check_engine(PARQUET_ENGINE)


def find_year_bounds(root: Path) -> Tuple[int, int]:
    """Find min/max YYYY from folders anywhere under root."""
    years: List[int] = []
    for p in root.rglob("*"):
        if p.is_dir() and p.name.isdigit() and len(p.name) == 4:
            y = int(p.name)
            if 1900 <= y <= 3000:
                years.append(y)
    if not years:
        raise RuntimeError(f"No year-like folders found under {root}")
    return min(years), max(years)

def daily_index_from_years(ymin: int, ymax: int) -> pd.DatetimeIndex:
    """Daily index spanning Jan 1 (ymin) .. Dec 31 (ymax)."""
    start = pd.Timestamp(year=ymin, month=1, day=1)
    end   = pd.Timestamp(year=ymax, month=12, day=31)
    return pd.date_range(start=start, end=end, freq="D")

def all_parquet_files(root: Path) -> Iterable[Path]:
    """Yield every .parquet under root (recursive)."""
    return (p for p in root.rglob("*.parquet") if p.is_file())


def detect_date_column(df: pd.DataFrame) -> str:
    """Guess the date/timestamp column."""
    lower = {c.lower(): c for c in df.columns}
    for cand in DATE_CANDIDATES:
        if cand in lower:
            return lower[cand]
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        lc = c.lower()
        if any(t in lc for t in ("date", "data", "dia", "time", "ts")):
            try:
                pd.to_datetime(df[c], errors="raise")
                return c
            except Exception:
                pass
    raise ValueError(f"Could not find a date/time column in: {list(df.columns)}")


# ---------- robust indicator key helpers ----------
def _is_year(name: str) -> bool:
    return name.isdigit() and len(name) == 4 and 1900 <= int(name) <= 3000

def _is_month(name: str) -> bool:
    return name.isdigit() and len(name) in (1, 2) and 1 <= int(name) <= 12

def _find_parquet_anchor(p: Path) -> int:
    """Index of the last 'parquet' in path.parts; -1 if not found."""
    parts = p.parts
    idxs = [i for i, s in enumerate(parts) if s.lower() == "parquet"]
    return idxs[-1] if idxs else -1

def _indicator_key(path: Path) -> str:
    """
    Key = '<group>_-_<indicator>'
    - group: directory immediately under 'parquet'
    - indicator: deepest non-(year/month) directory below group; if none, use file stem
    """
    parts = path.parts
    i = _find_parquet_anchor(path)
    if i == -1 or i + 1 >= len(parts):
        # fallback
        group = parts[-2] if len(parts) >= 2 else "parquet"
        indicator = path.stem
        return f"{sanitize(group)}_-_{sanitize(indicator)}"

    group = parts[i + 1]
    mid = parts[i + 2 : -1]
    non_ym = [s for s in mid if not (_is_year(s) or _is_month(s))]
    indicator = non_ym[-1] if non_ym else path.stem
    return f"{sanitize(group)}_-_{sanitize(indicator)}"
# ---------------------------------------------------


def read_one_parquet(path: Path) -> pd.DataFrame:
    """Read one parquet as a DataFrame indexed by date, sorted ascending."""
    df = pd.read_parquet(path, engine=PARQUET_ENGINE)
    dcol = detect_date_column(df)
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).set_index(dcol).sort_index()
    return df

def rename_with_indicator(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Rename columns to '{indicator_key}_-_{original_col}'."""
    key = _indicator_key(path)
    return df.rename(columns=lambda c: f"{key}_-_{sanitize(c)}")


def build_X(root: Path) -> pd.DataFrame:
    ymin, ymax = find_year_bounds(root)
    idx = daily_index_from_years(ymin, ymax)

    parts: List[pd.DataFrame] = []
    for f in sorted(all_parquet_files(root)):
        try:
            df = read_one_parquet(f)
            df = df.loc[~df.index.duplicated(keep="last")]
            df = rename_with_indicator(df, f)
            df = df.reindex(idx)
            parts.append(df)
            print(f"[ok] {f}")
        except Exception as e:
            print(f"[skip] {f} -> {e}")

    if not parts:
        raise RuntimeError(f"No usable parquet files found under {root}")

    X = pd.concat(parts, axis=1, join="outer", copy=False)
    X = X.T.groupby(level=0).first().T
    X.index.name = "date"
    return X


if __name__ == "__main__":
    
    ROOT   = Path("data/00--raw/parquet")
    XTUDAO = Path("data/01--grouped/X_tudao.parquet")

    try:
        X = pd.read_parquet(XTUDAO)
    except FileNotFoundError:
        os.makedirs(XTUDAO.parent, exist_ok=True)
        X = build_X(ROOT)
    
    print(X.shape, "rows x cols")
    X.to_parquet(XTUDAO)

