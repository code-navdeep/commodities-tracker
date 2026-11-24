"""The store — every byte of SQLite in the app lives here.

Raw facts only: no generated columns, no derived values (those belong to
analytics). Each write batch bumps a revision counter in the meta table;
analytics memoize on that revision, which makes cache invalidation exact —
data changed → revision changed → old cache keys never match again.

Writes are upserts on natural keys. Nothing here updates or deletes user
data beyond that.
"""

import sqlite3
from contextlib import contextmanager
from typing import Iterable

import pandas as pd

from .domain import (CotWeek, DeliveryNotice, FuturesBar, InventoryPoint,
                     OptionChain, PriceSeries)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- CFTC Disaggregated categories as published. commercial = producer + swap
-- and non-commercial = managed money + other reportables are DERIVED in
-- analytics/features.py: storing those sums here would persist a computed
-- value and would discard the producer-vs-swap-dealer distinction, which is
-- the most informative split in the report.
CREATE TABLE IF NOT EXISTS cot (
    asset TEXT NOT NULL,
    report_date TEXT NOT NULL,
    prod_long INTEGER NOT NULL,
    prod_short INTEGER NOT NULL,
    swap_long INTEGER NOT NULL,
    swap_short INTEGER NOT NULL,
    swap_spreading INTEGER NOT NULL,
    mm_long INTEGER NOT NULL,
    mm_short INTEGER NOT NULL,
    mm_spreading INTEGER NOT NULL,
    other_long INTEGER NOT NULL,
    other_short INTEGER NOT NULL,
    other_spreading INTEGER NOT NULL,
    nonreportable_long INTEGER NOT NULL,
    nonreportable_short INTEGER NOT NULL,
    open_interest INTEGER NOT NULL,
    PRIMARY KEY (asset, report_date)
);

CREATE TABLE IF NOT EXISTS delivery_reports (
    id INTEGER PRIMARY KEY,
    asset TEXT NOT NULL,
    intent_date TEXT NOT NULL,
    delivery_date TEXT NOT NULL,
    contract_month TEXT NOT NULL,
    total_issued INTEGER NOT NULL,
    total_stopped INTEGER NOT NULL,
    month_to_date INTEGER NOT NULL,
    settlement REAL,
    UNIQUE (asset, intent_date, contract_month)
);

CREATE TABLE IF NOT EXISTS delivery_firms (
    report_id INTEGER NOT NULL REFERENCES delivery_reports(id) ON DELETE CASCADE,
    firm_code TEXT NOT NULL,
    org_code TEXT NOT NULL,
    firm_name TEXT NOT NULL,
    issued INTEGER NOT NULL,
    stopped INTEGER NOT NULL,
    -- House vs customer accounts of one firm are distinct economic actors:
    -- both rows are kept, distinguished by org_code.
    UNIQUE (report_id, firm_code, org_code)
);

-- future_close is the front contract's SETTLE (the name predates the rest of
-- the bar and is kept because analytics, views and tests all reference it).
-- The other future_* columns are the same settlement row's remaining fields,
-- which arrive free in a payload already being parsed; future_high/_low are
-- what Parkinson realized volatility is computed from.
CREATE TABLE IF NOT EXISTS prices (
    asset TEXT NOT NULL,
    date TEXT NOT NULL,
    spot REAL,
    future_close REAL,
    future_month TEXT,
    future_open REAL,
    future_high REAL,
    future_low REAL,
    future_last REAL,
    future_volume INTEGER,
    future_oi INTEGER,
    usd_index REAL,
    vix REAL,
    PRIMARY KEY (asset, date)
);

CREATE TABLE IF NOT EXISTS inventory (
    asset TEXT NOT NULL,
    series TEXT NOT NULL,
    period TEXT NOT NULL,
    value REAL NOT NULL,
    units TEXT NOT NULL,
    PRIMARY KEY (asset, series, period)
);

CREATE TABLE IF NOT EXISTS options (
    asset TEXT NOT NULL,
    expiry_month TEXT NOT NULL,
    data_date TEXT NOT NULL,
    strike REAL NOT NULL,
    side TEXT NOT NULL,
    settle REAL,
    open REAL,
    high REAL,
    low REAL,
    volume INTEGER NOT NULL,
    open_interest INTEGER NOT NULL,
    underlying REAL,
    rate REAL NOT NULL,
    expiry_date TEXT,
    PRIMARY KEY (asset, expiry_month, data_date, strike, side)
);

CREATE INDEX IF NOT EXISTS idx_delivery_reports_asset_date
    ON delivery_reports(asset, intent_date);
CREATE INDEX IF NOT EXISTS idx_options_chain
    ON options(asset, expiry_month, data_date);
"""

# Columns whose fetched series may create new price rows. Macro columns only
# fill rows that already exist — a date carrying just DXY/VIX and no price is
# analytical noise.
_ROW_CREATING_PRICE_COLUMNS = {'spot'}
_PRICE_COLUMNS = ['spot', 'usd_index', 'vix']

# Written only by put_futures, from one FuturesBar per trade date — futures
# never travel through put_prices, so each column has exactly one write path.
_FUTURES_COLUMNS = ['future_month', 'future_open', 'future_high', 'future_low',
                    'future_last', 'future_close', 'future_volume', 'future_oi']


class Store:
    def __init__(self, db_path):
        self.db_path = str(db_path)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init(self):
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('revision', '0')"
            )

    # -- revision ------------------------------------------------------------

    def revision(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='revision'").fetchone()
            return int(row[0]) if row else 0

    @staticmethod
    def _bump(conn):
        conn.execute(
            "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key='revision'"
        )

    # -- writes (one revision bump per batch) --------------------------------

    def put_cot(self, weeks: Iterable[CotWeek]) -> int:
        weeks = list(weeks)
        if not weeks:
            return 0
        with self._conn() as conn:
            for w in weeks:
                conn.execute("""
                    INSERT INTO cot VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(asset, report_date) DO UPDATE SET
                        prod_long=excluded.prod_long, prod_short=excluded.prod_short,
                        swap_long=excluded.swap_long, swap_short=excluded.swap_short,
                        swap_spreading=excluded.swap_spreading,
                        mm_long=excluded.mm_long, mm_short=excluded.mm_short,
                        mm_spreading=excluded.mm_spreading,
                        other_long=excluded.other_long, other_short=excluded.other_short,
                        other_spreading=excluded.other_spreading,
                        nonreportable_long=excluded.nonreportable_long,
                        nonreportable_short=excluded.nonreportable_short,
                        open_interest=excluded.open_interest
                """, (w.asset, w.report_date,
                      w.prod_long, w.prod_short,
                      w.swap_long, w.swap_short, w.swap_spreading,
                      w.mm_long, w.mm_short, w.mm_spreading,
                      w.other_long, w.other_short, w.other_spreading,
                      w.nonreportable_long, w.nonreportable_short, w.open_interest))
            self._bump(conn)
        return len(weeks)

    def put_futures(self, bars: Iterable[FuturesBar]) -> int:
        """Upsert front-month futures bars. Creates price rows (as the settle
        alone used to), overwriting only the future_* columns and never
        touching spot or the macro columns."""
        bars = list(bars)
        if not bars:
            return 0
        assignments = ', '.join(f"{c}=?" for c in _FUTURES_COLUMNS)
        placeholders = ', '.join('?' for _ in _FUTURES_COLUMNS)
        with self._conn() as conn:
            for b in bars:
                values = (b.contract_month, b.open, b.high, b.low,
                          b.last, b.settle, b.volume, b.open_interest)
                exists = conn.execute(
                    "SELECT 1 FROM prices WHERE asset=? AND date=?",
                    (b.asset, b.date)).fetchone()
                if exists:
                    conn.execute(
                        f"UPDATE prices SET {assignments} WHERE asset=? AND date=?",
                        (*values, b.asset, b.date))
                else:
                    conn.execute(
                        f"INSERT INTO prices (asset, date, {', '.join(_FUTURES_COLUMNS)}) "
                        f"VALUES (?, ?, {placeholders})",
                        (b.asset, b.date, *values))
            self._bump(conn)
        return len(bars)

    def put_delivery(self, notices: Iterable[DeliveryNotice]) -> int:
        """Insert notices that don't exist yet (natural key: asset + intent
        date + contract month). Existing reports are left untouched."""
        notices = list(notices)
        if not notices:
            return 0
        written = 0
        with self._conn() as conn:
            for n in notices:
                exists = conn.execute(
                    "SELECT 1 FROM delivery_reports WHERE asset=? AND intent_date=? AND contract_month=?",
                    (n.asset, n.intent_date, n.contract_month)
                ).fetchone()
                if exists:
                    continue
                cursor = conn.execute("""
                    INSERT INTO delivery_reports
                        (asset, intent_date, delivery_date, contract_month,
                         total_issued, total_stopped, month_to_date, settlement)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (n.asset, n.intent_date, n.delivery_date, n.contract_month,
                      n.total_issued, n.total_stopped, n.month_to_date, n.settlement))
                report_id = cursor.lastrowid
                for f in n.firms:
                    conn.execute("""
                        INSERT INTO delivery_firms
                            (report_id, firm_code, org_code, firm_name, issued, stopped)
                        VALUES (?,?,?,?,?,?)
                    """, (report_id, f.firm_code, f.org_code, f.firm_name, f.issued, f.stopped))
                written += 1
            if written:
                self._bump(conn)
        return written

    def put_prices(self, series_list: Iterable[PriceSeries]) -> int:
        """Upsert price columns. spot/future_close create rows; macro columns
        only fill rows that already exist. Fetched values overwrite their own
        column and never touch the others. Returns cells written."""
        series_list = list(series_list)
        if not series_list:
            return 0
        written = 0
        with self._conn() as conn:
            for series in series_list:
                if series.column not in _PRICE_COLUMNS:
                    raise ValueError(f"Unknown price column '{series.column}'")
                create_rows = series.column in _ROW_CREATING_PRICE_COLUMNS
                for date, value in sorted(series.values.items()):
                    exists = conn.execute(
                        "SELECT 1 FROM prices WHERE asset=? AND date=?",
                        (series.asset, date)
                    ).fetchone()
                    if exists:
                        conn.execute(
                            f"UPDATE prices SET {series.column}=? WHERE asset=? AND date=?",
                            (value, series.asset, date))
                        written += 1
                    elif create_rows:
                        conn.execute(
                            f"INSERT INTO prices (asset, date, {series.column}) VALUES (?,?,?)",
                            (series.asset, date, value))
                        written += 1
            if written:
                self._bump(conn)
        return written

    def put_inventory(self, points: Iterable[InventoryPoint]) -> int:
        points = list(points)
        if not points:
            return 0
        with self._conn() as conn:
            for p in points:
                conn.execute("""
                    INSERT INTO inventory VALUES (?,?,?,?,?)
                    ON CONFLICT(asset, series, period) DO UPDATE SET
                        value=excluded.value, units=excluded.units
                """, (p.asset, p.series, p.period, p.value, p.units))
            self._bump(conn)
        return len(points)

    def put_chain(self, chain: OptionChain) -> int:
        if not chain.quotes:
            return 0
        with self._conn() as conn:
            for q in chain.quotes:
                conn.execute("""
                    INSERT OR REPLACE INTO options
                        (asset, expiry_month, data_date, strike, side,
                         settle, open, high, low, volume, open_interest,
                         underlying, rate, expiry_date)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (chain.asset, chain.expiry_month, chain.data_date, q.strike, q.side,
                      q.settle, q.open, q.high, q.low, q.volume, q.open_interest,
                      chain.underlying, chain.rate, chain.expiry_date))
            self._bump(conn)
        return len(chain.quotes)

    # -- high-water reads (drive incremental fetching) -----------------------

    def latest_cot_date(self, asset: str):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(report_date) FROM cot WHERE asset=?", (asset,)).fetchone()
            return row[0]

    def latest_price_date(self, asset: str, column: str):
        if column not in _PRICE_COLUMNS:
            raise ValueError(f"Unknown price column '{column}'")
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT MAX(date) FROM prices WHERE asset=? AND {column} IS NOT NULL",
                (asset,)).fetchone()
            return row[0]

    def latest_inventory_period(self, asset: str, series: str):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(period) FROM inventory WHERE asset=? AND series=?",
                (asset, series)).fetchone()
            return row[0]

    def has_chain(self, asset: str, expiry_month: str, data_date: str) -> bool:
        with self._conn() as conn:
            return conn.execute(
                "SELECT 1 FROM options WHERE asset=? AND expiry_month=? AND data_date=? LIMIT 1",
                (asset, expiry_month, data_date)).fetchone() is not None

    # -- frame reads (what analytics and views consume) ----------------------

    def cot(self, asset: str) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM cot WHERE asset=? ORDER BY report_date",
                conn, params=(asset,))

    def delivery(self, asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(reports, firm detail) for one asset, chronological."""
        with self._conn() as conn:
            reports = pd.read_sql_query(
                "SELECT * FROM delivery_reports WHERE asset=? ORDER BY intent_date",
                conn, params=(asset,))
            firms = pd.read_sql_query("""
                SELECT f.* FROM delivery_firms f
                JOIN delivery_reports r ON r.id = f.report_id
                WHERE r.asset=?
            """, conn, params=(asset,))
        return reports, firms

    def prices(self, asset: str) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM prices WHERE asset=? ORDER BY date",
                conn, params=(asset,))

    def inventory(self, asset: str, series: str = None) -> pd.DataFrame:
        """Weekly inventory rows for one asset, chronological. `series` limits
        to a single registry short name."""
        query = "SELECT * FROM inventory WHERE asset=?"
        params = [asset]
        if series:
            query += " AND series=?"
            params.append(series)
        with self._conn() as conn:
            return pd.read_sql_query(query + " ORDER BY period", conn, params=params)

    def chain(self, asset: str, expiry_month: str, data_date: str) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query("""
                SELECT * FROM options
                WHERE asset=? AND expiry_month=? AND data_date=?
                ORDER BY strike, side
            """, conn, params=(asset, expiry_month, data_date))

    def chain_dates(self, asset: str, expiry_month: str) -> list:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT DISTINCT data_date FROM options
                WHERE asset=? AND expiry_month=? ORDER BY data_date DESC
            """, (asset, expiry_month)).fetchall()
            return [r[0] for r in rows]

    def option_dates(self, asset: str) -> list:
        """Every trade date with a stored chain, ascending — the axis the
        options timeseries is built on."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT data_date FROM options WHERE asset=? ORDER BY data_date",
                (asset,)).fetchall()
            return [r[0] for r in rows]

    def chains_on(self, asset: str, data_date: str) -> pd.DataFrame:
        """Every expiry's chain for one trade date."""
        with self._conn() as conn:
            return pd.read_sql_query("""
                SELECT * FROM options WHERE asset=? AND data_date=?
                ORDER BY expiry_month, strike, side
            """, conn, params=(asset, data_date))

    def expiries(self, asset: str) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT expiry_month FROM options WHERE asset=? ORDER BY expiry_month",
                (asset,)).fetchall()
            return [r[0] for r in rows]

    def option_rates(self, asset: str) -> pd.DataFrame:
        """Average stored risk-free rate per data date — feeds the
        carry-adjusted basis feature."""
        with self._conn() as conn:
            return pd.read_sql_query("""
                SELECT data_date AS date, AVG(rate) AS rate
                FROM options WHERE asset=? GROUP BY data_date
            """, conn, params=(asset,))

    def stream_presence(self) -> dict:
        """{table_kind: set(assets with rows)} — drives UI enablement."""
        presence = {}
        with self._conn() as conn:
            for kind, table in [('delivery', 'delivery_reports'), ('prices', 'prices'),
                                ('cot', 'cot'), ('inventory', 'inventory'),
                                ('options', 'options')]:
                rows = conn.execute(f"SELECT DISTINCT asset FROM {table}").fetchall()
                presence[kind] = {r[0] for r in rows}
        return presence
