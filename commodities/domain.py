"""Typed records — the shapes data takes at every boundary.

Sources produce these; the store persists them; refresh reports summarize
them. All dates are ISO strings ('YYYY-MM-DD'); expiry months are 'YYYY-MM'.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CotWeek:
    """One CFTC Disaggregated week, as published — the four reportable trader
    categories kept separate. Producer/merchant and swap dealers are both
    'commercial' in the legacy report but frequently mean opposite things
    (a miner selling forward vs a bank hedging investor length), so the
    commercial/non-commercial aggregation happens in analytics/features.py,
    not here."""
    asset: str
    report_date: str
    prod_long: int               # producer/merchant/processor/user
    prod_short: int              # (this category has no spreading field)
    swap_long: int
    swap_short: int
    swap_spreading: int
    mm_long: int                 # managed money
    mm_short: int
    mm_spreading: int
    other_long: int              # other reportables
    other_short: int
    other_spreading: int
    nonreportable_long: int
    nonreportable_short: int
    open_interest: int


@dataclass(frozen=True)
class FirmLine:
    firm_code: str
    org_code: str        # 'H' house / 'C' customer — distinct economic actors
    firm_name: str
    issued: int
    stopped: int


@dataclass(frozen=True)
class DeliveryNotice:
    asset: str
    contract_month: str          # 'YYYY-MM'
    intent_date: str
    delivery_date: str
    total_issued: int
    total_stopped: int
    month_to_date: int
    settlement: float | None     # contract settlement price printed on the notice
    firms: tuple[FirmLine, ...]


@dataclass(frozen=True)
class PriceSeries:
    """One column of price_data for one asset: {iso_date: value}. Futures no
    longer travel this way — they carry a whole bar (see FuturesBar)."""
    asset: str
    column: str                  # 'spot' | 'usd_index' | 'vix'
    values: dict


@dataclass(frozen=True)
class FuturesBar:
    """One trade date's front-month futures settlement row, as CME publishes
    it. `change` is deliberately not carried: it is settle minus the prior
    settle, i.e. derived, and the store holds facts only."""
    asset: str
    date: str                    # trade date (ISO)
    contract_month: str          # 'YYYY-MM' of the front contract
    open: float | None
    high: float | None
    low: float | None
    last: float | None           # last traded price (not the settle)
    settle: float | None
    volume: int
    open_interest: int


@dataclass(frozen=True)
class InventoryPoint:
    """One weekly EIA observation: a stock level (or, for production, a flow)
    as published, with the units EIA reports it in."""
    asset: str
    series: str          # registry short name, e.g. 'cushing'
    period: str          # ISO week-ending date
    value: float
    units: str           # 'MBBL' | 'BCF' | 'MBBL/D'


@dataclass(frozen=True)
class OptionQuote:
    strike: float
    side: str                    # 'Call' | 'Put'
    settle: float | None
    open: float | None
    high: float | None
    low: float | None
    volume: int
    open_interest: int


@dataclass(frozen=True)
class OptionChain:
    asset: str
    expiry_month: str            # 'YYYY-MM'
    expiry_date: str | None      # option's last-trade date (ISO)
    data_date: str               # trade date of this settlement snapshot
    underlying: float | None     # matching futures month's settle
    rate: float                  # decimal risk-free rate (3M T-bill)
    quotes: tuple[OptionQuote, ...]


@dataclass
class StreamResult:
    """Outcome of refreshing one stream."""
    ok: bool = True
    counts: dict = field(default_factory=dict)    # {asset: records written}
    errors: list = field(default_factory=list)    # human-readable strings


@dataclass
class RefreshReport:
    """Outcome of one refresh run, keyed by stream name."""
    streams: dict = field(default_factory=dict)   # {stream: StreamResult}

    @property
    def all_ok(self) -> bool:
        return all(result.ok for result in self.streams.values())
