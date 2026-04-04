"""
Persist / restore PythonSessionTool kernel state (DataFrames + metadata) for resume.

Writes under run_output_dir/kernel_snapshot/: meta.json + optional df_*.parquet.
Requires pyarrow for Parquet (see requirements.txt).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

KERNEL_SNAPSHOT_SUBDIR = "kernel_snapshot"


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars and nested structures for json.dumps."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = _make_json_serializable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(x) for x in obj]
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return _make_json_serializable(obj.item())
        if isinstance(obj, np.ndarray):
            return _make_json_serializable(obj.tolist())
    except ImportError:
        pass
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "isoformat") and callable(getattr(obj, "isoformat")):
        try:
            return obj.isoformat()
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)

_META_LIST_KEYS = (
    "DATASET_COLUMNS",
    "NUMERIC_COLUMNS",
    "CATEGORICAL_COLUMNS",
    "BOOLEAN_COLUMNS",
    "ORIGINAL_NUMERIC_COLUMNS",
    "ORIGINAL_CATEGORICAL_COLUMNS",
)


def kernel_snapshot_dir(run_dir: Path) -> Path:
    return Path(run_dir).resolve() / KERNEL_SNAPSHOT_SUBDIR


def _snapshot_enabled() -> bool:
    v = os.getenv("SESSION_SNAPSHOT", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def save_kernel_snapshot(executor: Any, run_dir: Path) -> None:
    """Persist df_raw / df_clean / df_features and JSON-serializable globals into kernel_snapshot/."""
    if not _snapshot_enabled():
        return

    snap = kernel_snapshot_dir(run_dir)
    snap.mkdir(parents=True, exist_ok=True)
    g = executor.session_globals

    meta: Dict[str, Any] = {
        "schema": 1,
        "dataset_path": str(g.get("DATASET_PATH") or ""),
        "TIME_INDEX_OK": bool(g.get("TIME_INDEX_OK")),
        "validation_report": (
            g.get("validation_report")
            if isinstance(g.get("validation_report"), (dict, list))
            else []
        ),
    }
    for k in _META_LIST_KEYS:
        v = g.get(k)
        if isinstance(v, list):
            meta[k] = [str(x) for x in v]
        elif v is not None:
            meta[k] = list(v) if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)) else v

    raw = g.get("df_raw")
    if isinstance(raw, pd.DataFrame):
        meta["DATASET_SHAPE"] = [int(raw.shape[0]), int(raw.shape[1])]
    else:
        meta["DATASET_SHAPE"] = None

    for name in ("df_raw", "df_clean", "df_features"):
        df = g.get(name)
        path = snap / f"{name}.parquet"
        if isinstance(df, pd.DataFrame):
            try:
                df.to_parquet(path, index=False)
            except Exception as e:
                print(f"[SESSION_SNAPSHOT] Parquet write failed ({name}): {e}")
                meta[f"{name}_saved"] = False
                continue
            meta[f"{name}_saved"] = True
        else:
            meta[f"{name}_saved"] = False
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

    meta_safe = _make_json_serializable(meta)
    (snap / "meta.json").write_text(
        json.dumps(meta_safe, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def try_load_kernel_snapshot(executor: Any, run_dir: Path) -> bool:
    """If kernel_snapshot/meta.json exists, hydrate session_globals. Returns True if loaded."""
    snap = kernel_snapshot_dir(run_dir)
    meta_path = snap / "meta.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    g = executor.session_globals

    try:
        for name in ("df_raw", "df_clean", "df_features"):
            key_saved = meta.get(f"{name}_saved")
            path = snap / f"{name}.parquet"
            if key_saved is True and path.is_file():
                g[name] = pd.read_parquet(path)
            elif path.is_file():
                g[name] = pd.read_parquet(path)
            else:
                g[name] = None

        g["TIME_INDEX_OK"] = bool(meta.get("TIME_INDEX_OK", False))
        vr = meta.get("validation_report")
        if isinstance(vr, (dict, list)):
            g["validation_report"] = vr
        elif vr is not None:
            g["validation_report"] = []

        dp = meta.get("dataset_path")
        if isinstance(dp, str) and dp:
            g["DATASET_PATH"] = dp

        for k in _META_LIST_KEYS:
            if k in meta and isinstance(meta[k], list):
                g[k] = list(meta[k])

        raw = g.get("df_raw")
        if isinstance(raw, pd.DataFrame):
            g["DATASET_SHAPE"] = (int(raw.shape[0]), int(raw.shape[1]))
        elif isinstance(meta.get("DATASET_SHAPE"), list) and len(meta["DATASET_SHAPE"]) == 2:
            g["DATASET_SHAPE"] = (int(meta["DATASET_SHAPE"][0]), int(meta["DATASET_SHAPE"][1]))
    except Exception as e:
        print(f"[SESSION_SNAPSHOT] Load failed: {e}")
        return False

    return True
