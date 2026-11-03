"""SQLite persistence layer for the pothole verification dashboard.

Schema:
  videos       — registered video files
  events       — candidate pothole events (one row per CandidateEvent)
  detections   — individual frame-level detector observations per event
  reviews      — human review decisions (FK → events)

Design rules:
  - Video files, evidence images, and clips are stored as filesystem paths only.
  - Model predictions are never overwritten by human corrections; both are preserved.
  - All timestamps are stored in milliseconds (integer).
  - UUIDs are stored as TEXT.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS videos (
    video_id            TEXT PRIMARY KEY,
    filename            TEXT NOT NULL,
    filepath            TEXT NOT NULL,
    road_segment_id     TEXT NOT NULL DEFAULT 'seg_001',
    duration_sec        REAL,
    fps                 REAL,
    width               INTEGER,
    height              INTEGER,
    file_size_mb        REAL,
    registered_at       TEXT NOT NULL,
    processing_status   TEXT NOT NULL DEFAULT 'pending',
    processing_started  TEXT,
    processing_finished TEXT,
    processing_error    TEXT,
    sample_fps          REAL,
    model_path          TEXT,
    model_version       TEXT,
    detector_conf       REAL,
    detector_iou        REAL,
    total_frames_sampled INTEGER,
    notes               TEXT
);

CREATE TABLE IF NOT EXISTS events (
    event_id            TEXT PRIMARY KEY,
    video_id            TEXT NOT NULL REFERENCES videos(video_id),
    road_segment_id     TEXT NOT NULL DEFAULT 'seg_001',
    track_id            INTEGER,
    first_timestamp_ms  INTEGER NOT NULL,
    last_timestamp_ms   INTEGER NOT NULL,
    first_frame_index   INTEGER,
    last_frame_index    INTEGER,
    duration_ms         INTEGER,
    observation_count   INTEGER NOT NULL DEFAULT 0,
    detector_conf_max   REAL,
    detector_conf_mean  REAL,
    rep_x1              INTEGER,
    rep_y1              INTEGER,
    rep_x2              INTEGER,
    rep_y2              INTEGER,
    full_frame_path     TEXT,
    crop_path           TEXT,
    clip_path           TEXT,
    tracker_config      TEXT,          -- JSON blob
    policy_a_retained   INTEGER NOT NULL DEFAULT 1,
    policy_b_retained   INTEGER NOT NULL DEFAULT 0,
    policy_c_score      REAL,
    policy_c_retained   INTEGER,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id            TEXT NOT NULL REFERENCES events(event_id),
    video_id            TEXT NOT NULL,
    frame_index         INTEGER NOT NULL,
    timestamp_ms        INTEGER NOT NULL,
    x1                  INTEGER NOT NULL,
    y1                  INTEGER NOT NULL,
    x2                  INTEGER NOT NULL,
    y2                  INTEGER NOT NULL,
    conf                REAL NOT NULL,
    laplacian_var       REAL
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id            TEXT NOT NULL REFERENCES events(event_id),
    decision            TEXT NOT NULL,   -- confirmed | rejected | uncertain | flag_grouping
    fp_category         TEXT,            -- shadow | repaired_patch | manhole | ordinary_road | other
    reviewer_notes      TEXT,
    reviewed_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_video_id ON events(video_id);
CREATE INDEX IF NOT EXISTS idx_events_first_ts ON events(first_timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_detections_event_id ON detections(event_id);
CREATE INDEX IF NOT EXISTS idx_reviews_event_id ON reviews(event_id);
"""

# ---------------------------------------------------------------------------
# Database connection manager
# ---------------------------------------------------------------------------

class Database:
    """Thin wrapper around a SQLite connection."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA_SQL)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Video registration
    # ------------------------------------------------------------------

    def register_video(self, video_id: str, meta: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO videos
                  (video_id, filename, filepath, road_segment_id, duration_sec,
                   fps, width, height, file_size_mb, registered_at,
                   processing_status, notes)
                VALUES
                  (:video_id, :filename, :filepath, :road_segment_id, :duration_sec,
                   :fps, :width, :height, :file_size_mb, :registered_at,
                   :processing_status, :notes)
                """,
                {
                    "video_id": video_id,
                    "filename": meta.get("filename", ""),
                    "filepath": meta.get("filepath", ""),
                    "road_segment_id": meta.get("road_segment_id", "seg_001"),
                    "duration_sec": meta.get("duration_sec"),
                    "fps": meta.get("fps"),
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                    "file_size_mb": meta.get("file_size_mb"),
                    "registered_at": self._now(),
                    "processing_status": "pending",
                    "notes": meta.get("notes", ""),
                },
            )

    def mark_processing_started(self, video_id: str, run_config: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE videos SET
                    processing_status   = 'processing',
                    processing_started  = :ts,
                    sample_fps          = :sample_fps,
                    model_path          = :model_path,
                    detector_conf       = :detector_conf,
                    detector_iou        = :detector_iou
                WHERE video_id = :video_id
                """,
                {
                    "video_id": video_id,
                    "ts": self._now(),
                    "sample_fps": run_config.get("sample_fps"),
                    "model_path": run_config.get("model_path"),
                    "detector_conf": run_config.get("detector_conf"),
                    "detector_iou": run_config.get("detector_iou"),
                },
            )

    def mark_processing_done(self, video_id: str, frames_sampled: int) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE videos SET
                    processing_status       = 'done',
                    processing_finished     = :ts,
                    total_frames_sampled    = :frames
                WHERE video_id = :video_id
                """,
                {"video_id": video_id, "ts": self._now(), "frames": frames_sampled},
            )

    def mark_processing_error(self, video_id: str, error: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE videos SET
                    processing_status = 'error',
                    processing_error  = :error,
                    processing_finished = :ts
                WHERE video_id = :video_id
                """,
                {"video_id": video_id, "error": error, "ts": self._now()},
            )

    # ------------------------------------------------------------------
    # Event persistence
    # ------------------------------------------------------------------

    def save_events(self, events: list) -> None:
        """Insert or replace CandidateEvent objects."""
        from src.tracking.event_grouper import CandidateEvent

        with self._conn() as conn:
            for ev in events:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO events
                      (event_id, video_id, road_segment_id, track_id,
                       first_timestamp_ms, last_timestamp_ms,
                       first_frame_index, last_frame_index, duration_ms,
                       observation_count, detector_conf_max, detector_conf_mean,
                       rep_x1, rep_y1, rep_x2, rep_y2,
                       full_frame_path, crop_path, clip_path,
                       tracker_config, policy_a_retained, policy_b_retained,
                       policy_c_score, policy_c_retained, created_at)
                    VALUES
                      (:event_id, :video_id, :road_segment_id, :track_id,
                       :first_timestamp_ms, :last_timestamp_ms,
                       :first_frame_index, :last_frame_index, :duration_ms,
                       :observation_count, :detector_conf_max, :detector_conf_mean,
                       :rep_x1, :rep_y1, :rep_x2, :rep_y2,
                       :full_frame_path, :crop_path, :clip_path,
                       :tracker_config, :policy_a_retained, :policy_b_retained,
                       :policy_c_score, :policy_c_retained, :created_at)
                    """,
                    {
                        "event_id": ev.event_id,
                        "video_id": ev.video_id,
                        "road_segment_id": ev.road_segment_id,
                        "track_id": ev.track_id,
                        "first_timestamp_ms": ev.first_timestamp_ms,
                        "last_timestamp_ms": ev.last_timestamp_ms,
                        "first_frame_index": ev.first_frame_index,
                        "last_frame_index": ev.last_frame_index,
                        "duration_ms": ev.duration_ms,
                        "observation_count": ev.observation_count,
                        "detector_conf_max": ev.detector_conf_max,
                        "detector_conf_mean": ev.detector_conf_mean,
                        "rep_x1": ev.rep_x1,
                        "rep_y1": ev.rep_y1,
                        "rep_x2": ev.rep_x2,
                        "rep_y2": ev.rep_y2,
                        "full_frame_path": ev.full_frame_path,
                        "crop_path": ev.crop_path,
                        "clip_path": ev.clip_path,
                        "tracker_config": json.dumps(ev.tracker_config),
                        "policy_a_retained": int(ev.policy_a_retained),
                        "policy_b_retained": int(ev.policy_b_retained),
                        "policy_c_score": ev.policy_c_score,
                        "policy_c_retained": (
                            int(ev.policy_c_retained)
                            if ev.policy_c_retained is not None else None
                        ),
                        "created_at": self._now(),
                    },
                )
                # Save individual observations
                for obs in ev.observations:
                    conn.execute(
                        """
                        INSERT INTO detections
                          (event_id, video_id, frame_index, timestamp_ms,
                           x1, y1, x2, y2, conf, laplacian_var)
                        VALUES
                          (:event_id, :video_id, :frame_index, :timestamp_ms,
                           :x1, :y1, :x2, :y2, :conf, :laplacian_var)
                        """,
                        {
                            "event_id": ev.event_id,
                            "video_id": ev.video_id,
                            "frame_index": obs.frame_index,
                            "timestamp_ms": obs.timestamp_ms,
                            "x1": obs.x1, "y1": obs.y1,
                            "x2": obs.x2, "y2": obs.y2,
                            "conf": obs.conf,
                            "laplacian_var": obs.laplacian_var,
                        },
                    )

    # ------------------------------------------------------------------
    # Review persistence
    # ------------------------------------------------------------------

    def save_review(
        self,
        event_id: str,
        decision: str,
        fp_category: Optional[str] = None,
        notes: str = "",
    ) -> None:
        """Upsert a review decision.  Replaces any prior review for the same event."""
        with self._conn() as conn:
            # Delete prior review for this event (single-reviewer workflow)
            conn.execute("DELETE FROM reviews WHERE event_id = ?", (event_id,))
            conn.execute(
                """
                INSERT INTO reviews (event_id, decision, fp_category, reviewer_notes, reviewed_at)
                VALUES (:event_id, :decision, :fp_category, :notes, :ts)
                """,
                {
                    "event_id": event_id,
                    "decision": decision,
                    "fp_category": fp_category,
                    "notes": notes,
                    "ts": self._now(),
                },
            )

    def delete_review(self, event_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM reviews WHERE event_id = ?", (event_id,))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_videos(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM videos ORDER BY registered_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_video(self, video_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE video_id = ?", (video_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_events(
        self,
        video_id: Optional[str] = None,
        include_reviews: bool = True,
    ) -> list[dict]:
        with self._conn() as conn:
            if video_id:
                rows = conn.execute(
                    "SELECT * FROM events WHERE video_id = ? ORDER BY first_timestamp_ms",
                    (video_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY video_id, first_timestamp_ms"
                ).fetchall()

            events = [dict(r) for r in rows]

            if include_reviews:
                review_map = self._get_review_map(conn)
                for ev in events:
                    ev["review"] = review_map.get(ev["event_id"])

        return events

    def get_event(self, event_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if row is None:
                return None
            ev = dict(row)
            review_map = self._get_review_map(conn)
            ev["review"] = review_map.get(event_id)
            # Detections
            dets = conn.execute(
                "SELECT * FROM detections WHERE event_id = ? ORDER BY timestamp_ms",
                (event_id,),
            ).fetchall()
            ev["observations"] = [dict(d) for d in dets]
        return ev

    def _get_review_map(self, conn: sqlite3.Connection) -> dict[str, dict]:
        rows = conn.execute("SELECT * FROM reviews").fetchall()
        return {r["event_id"]: dict(r) for r in rows}

    def get_summary_stats(self) -> dict:
        with self._conn() as conn:
            total_videos = conn.execute(
                "SELECT COUNT(*) FROM videos WHERE processing_status = 'done'"
            ).fetchone()[0]
            total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            policy_a = conn.execute(
                "SELECT COUNT(*) FROM events WHERE policy_a_retained = 1"
            ).fetchone()[0]
            policy_b = conn.execute(
                "SELECT COUNT(*) FROM events WHERE policy_b_retained = 1"
            ).fetchone()[0]
            reviewed = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
            confirmed = conn.execute(
                "SELECT COUNT(*) FROM reviews WHERE decision = 'confirmed'"
            ).fetchone()[0]
            rejected = conn.execute(
                "SELECT COUNT(*) FROM reviews WHERE decision = 'rejected'"
            ).fetchone()[0]
            uncertain = conn.execute(
                "SELECT COUNT(*) FROM reviews WHERE decision = 'uncertain'"
            ).fetchone()[0]
            total_detections = conn.execute(
                "SELECT COUNT(*) FROM detections"
            ).fetchone()[0]

        return {
            "total_videos": total_videos,
            "total_events": total_events,
            "policy_a_retained": policy_a,
            "policy_b_retained": policy_b,
            "reviewed": reviewed,
            "confirmed": confirmed,
            "rejected": rejected,
            "uncertain": uncertain,
            "total_detections": total_detections,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_events_csv(self, video_id: Optional[str] = None) -> str:
        """Return a CSV string of all events with review decisions."""
        import io, csv

        events = self.get_events(video_id=video_id, include_reviews=True)

        fields = [
            "event_id", "video_id", "road_segment_id",
            "first_timestamp_ms", "last_timestamp_ms", "duration_ms",
            "observation_count", "detector_conf_max", "detector_conf_mean",
            "rep_x1", "rep_y1", "rep_x2", "rep_y2",
            "policy_a_retained", "policy_b_retained",
            "policy_c_score", "policy_c_retained",
            "full_frame_path", "crop_path", "clip_path",
            "review_decision", "fp_category", "reviewer_notes",
            "model_version",
        ]

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()

        for ev in events:
            review = ev.get("review") or {}
            row = {
                **ev,
                "review_decision": review.get("decision", ""),
                "fp_category": review.get("fp_category", ""),
                "reviewer_notes": review.get("reviewer_notes", ""),
                "model_version": "",
            }
            writer.writerow(row)

        return buf.getvalue()

    def delete_events_for_video(self, video_id: str) -> None:
        """Remove all events (and their detections/reviews) for a video."""
        with self._conn() as conn:
            event_ids = [
                r[0] for r in conn.execute(
                    "SELECT event_id FROM events WHERE video_id = ?", (video_id,)
                ).fetchall()
            ]
            if event_ids:
                placeholders = ",".join("?" * len(event_ids))
                conn.execute(
                    f"DELETE FROM reviews WHERE event_id IN ({placeholders})", event_ids
                )
                conn.execute(
                    f"DELETE FROM detections WHERE event_id IN ({placeholders})", event_ids
                )
            conn.execute("DELETE FROM events WHERE video_id = ?", (video_id,))
