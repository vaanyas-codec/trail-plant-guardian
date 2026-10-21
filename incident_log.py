"""
Trail Plant Guardian: zone-based incident log (Objective 3).

An "incident" here is a user-submitted report: a photo of a host plant taken in a
trail zone, with the species confirmed by the user and a tick-box for visible damage.
The classifier cannot detect trampling itself, so 'damaged' is a user observation.

Storage is SQLite (standard library, free, one file).
"""
import random
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

DB_PATH = "incidents.db"

# --- Trail zones ------------------------------------------------------------
# PLACEHOLDERS: replace with your real trails/zones and coordinates you have checked.
ZONES = pd.DataFrame(
    [
        ("Z1", "Trail A", "Lower slope", 18.90, 73.30),
        ("Z2", "Trail A", "Ridge section", 18.91, 73.32),
        ("Z3", "Trail B", "Forest stretch", 19.10, 73.50),
        ("Z4", "Trail B", "Plateau", 19.12, 73.52),
        ("Z5", "Trail C", "Waterfall approach", 19.40, 73.70),
    ],
    columns=["zone_id", "trail", "zone_name", "lat", "lon"],
)

# --- Impact thresholds (arbitrary defaults; state them as such in your report) --
WINDOW_DAYS = 30   # only recent reports count
WATCH_N = 2        # damage reports in the window -> "Watch"
HIGH_N = 5         # damage reports in the window -> "High"


def _connect(db_path=DB_PATH):
    return sqlite3.connect(db_path)


def init_db(db_path=DB_PATH):
    with _connect(db_path) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                zone_id TEXT NOT NULL,
                species TEXT NOT NULL,          -- species confirmed by the user
                predicted TEXT,                 -- model's top-1 prediction
                confidence REAL,                -- model's top-1 probability
                tier TEXT,                      -- high / medium / low
                damaged INTEGER NOT NULL,       -- 1 if the user saw trampling/damage
                image_name TEXT,
                is_demo INTEGER NOT NULL DEFAULT 0   -- 1 = synthetic demo row
            )"""
        )


def log_incident(zone_id, species, damaged, predicted=None, confidence=None,
                 tier=None, image_name=None, is_demo=0, timestamp=None,
                 db_path=DB_PATH):
    init_db(db_path)
    ts = (timestamp or datetime.now()).isoformat(timespec="seconds")
    with _connect(db_path) as con:
        con.execute(
            """INSERT INTO incidents
               (timestamp, zone_id, species, predicted, confidence, tier, damaged, image_name, is_demo)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (ts, zone_id, species, predicted, confidence, tier, int(bool(damaged)),
             image_name, int(is_demo)),
        )


def load_incidents(db_path=DB_PATH, include_demo=True):
    init_db(db_path)
    with _connect(db_path) as con:
        df = pd.read_sql_query("SELECT * FROM incidents", con)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if not include_demo:
        df = df[df["is_demo"] == 0]
    return df


def zone_summary(df, zones=ZONES, hosts=None, window_days=WINDOW_DAYS,
                 high_n=HIGH_N, watch_n=WATCH_N, now=None):
    """One row per zone. `hosts` is an optional {plant: set(butterflies)} dict from the
    knowledge graph; it adds 'butterflies_at_risk' (hosts of plants reported damaged)."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now()
    recent = df[df["timestamp"] >= now - pd.Timedelta(days=window_days)]
    rows = []
    for z in zones.itertuples():
        zr = recent[recent["zone_id"] == z.zone_id]
        zd = zr[zr["damaged"] == 1]
        at_risk = set()
        if hosts:
            for sp in zd["species"].unique():
                at_risk |= set(hosts.get(sp, set()))
        n_dmg = len(zd)
        level = "High" if n_dmg >= high_n else "Watch" if n_dmg >= watch_n else "Low"
        top = zd["species"].value_counts().head(3)
        rows.append(
            dict(
                zone_id=z.zone_id, trail=z.trail, zone_name=z.zone_name,
                lat=z.lat, lon=z.lon,
                reports=len(zr), damaged_reports=n_dmg,
                species_affected=zd["species"].nunique(),
                top_species=", ".join(f"{s} ({c})" for s, c in top.items()),
                butterflies_at_risk=len(at_risk),
                last_report=zr["timestamp"].max() if len(zr) else pd.NaT,
                level=level,
            )
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["damaged_reports", "reports"], ascending=False).reset_index(drop=True)


def seed_demo(species_list, n=60, seed=42, zones=ZONES, db_path=DB_PATH):
    """Insert SYNTHETIC incidents (is_demo=1) so the dashboard has something to show.
    Always label this as simulated data in your report and demo."""
    rng = random.Random(seed)
    zone_ids = list(zones["zone_id"])
    weights = [5, 3, 1, 1, 2][: len(zone_ids)]  # skew so some zones stand out
    now = datetime.now()
    for _ in range(n):
        log_incident(
            zone_id=rng.choices(zone_ids, weights=weights)[0],
            species=rng.choice(species_list),
            damaged=rng.random() < 0.5,
            confidence=round(rng.uniform(0.3, 0.95), 2),
            tier="synthetic",
            is_demo=1,
            timestamp=now - timedelta(days=rng.uniform(0, 45)),
            db_path=db_path,
        )


def clear_demo(db_path=DB_PATH):
    init_db(db_path)
    with _connect(db_path) as con:
        con.execute("DELETE FROM incidents WHERE is_demo = 1")
