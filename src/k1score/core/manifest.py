"""One row per video, status of every ingest stage. Stored as parquet."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import INGEST_STAGES

STATUS_PENDING, STATUS_DONE, STATUS_FAILED = "pending", "done", "failed"

COLUMNS = (
    ["video_id", "url", "title"]
    + [f"status_{s}" for s in INGEST_STAGES]
    + ["schema_version", "n_frames", "valid_ratio", "error"]
)


def _clean(v):
    return None if (v is not None and not isinstance(v, str) and pd.isna(v)) else v


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            for rec in pd.read_parquet(self.path).to_dict("records"):
                rec = {k: _clean(v) for k, v in rec.items()}
                for c in COLUMNS:  # parquet written by an older version may lack newer stage columns
                    rec.setdefault(c, STATUS_PENDING if c.startswith("status_") else None)
                self.rows[rec["video_id"]] = rec

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(list(self.rows.values()), columns=COLUMNS)
        for col in ("n_frames",):
            df[col] = df[col].astype("Int64")
        df["valid_ratio"] = df["valid_ratio"].astype("float64")
        df.to_parquet(self.path, index=False)

    def get(self, video_id: str) -> dict | None:
        return self.rows.get(video_id)

    def upsert(self, video_id: str, **fields) -> dict:
        row = self.rows.setdefault(
            video_id,
            {c: (STATUS_PENDING if c.startswith("status_") else None) for c in COLUMNS} | {"video_id": video_id},
        )
        row.update(fields)
        self.save()
        return row

    def set_status(self, video_id: str, stage: str, status: str, error: str | None = None) -> None:
        self.upsert(video_id, **{f"status_{stage}": status, "error": error if status == STATUS_FAILED else None})

    def status(self, video_id: str, stage: str) -> str:
        row = self.rows.get(video_id)
        return row[f"status_{stage}"] if row else STATUS_PENDING

    def invalidate_from(self, video_id: str, stage: str) -> None:
        """Mark `stage` and everything after it as pending."""
        idx = INGEST_STAGES.index(stage)
        self.upsert(video_id, **{f"status_{s}": STATUS_PENDING for s in INGEST_STAGES[idx:]})

    def ids(self) -> list[str]:
        return sorted(self.rows)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.rows.values()), columns=COLUMNS)
