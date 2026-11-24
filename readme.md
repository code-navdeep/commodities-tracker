# Commodities Tracker — User Guide

## Read This First — the Three Boundaries of the Automation

Everything this app consumes is fetched by code — there is no manual data
entry anywhere, and no manual ingestion paths even exist. Exactly three
qualifiers bound that statement, and none of them are recurring work:

**1. Copper (HG) spot price is a genuine data hole — absent, not manual.**
There is no clean free source for a daily copper spot price: LME's cash
price is licensed, and the free aggregators are either bot-walled or
monthly-only. Nothing fetches HG spot, and nothing expects you to type it in
either — it is simply absent. What that costs you: HG's basis (futures −
spot) can't be computed, so its basis chart and basis-derived features stay
empty; everything else for copper (delivery, COT, futures close, options)
works fully. The moment a source is chosen, wiring it in is a one-line
`spot` entry in the `ASSETS` registry (`commodities/config.py`) plus a small
fetch function — the app is built to receive it.

**2. CL/NG spot needs the EIA API key — a one-time two-minute setup.**
WTI Cushing and Henry Hub daily spot come from the US EIA's official API,
which requires a free key. Until the key is set, every Prices refresh
skips CL/NG spot with an explicit log line — nothing breaks, but crude and
gas have no spot price, so their basis charts and any spot-return-based
analytics (including their ML panels, whose prediction target is the
forward spot return) stay off. Setup, once: register at
[eia.gov/opendata/register.php](https://www.eia.gov/opendata/register.php)
(free, instant, key arrives by email), copy `.env.example` in this folder
to a new file named `.env`, paste the key after `EIA_API_KEY=`, restart the
app. From then on CL/NG spot
backfills and stays current automatically — the key never needs touching
again. Note that EIA publishes these daily spot series with a few business
days' lag (the newest point is typically several days behind today), so
crude and gas basis trails the metals slightly; each refresh fills in
whatever has since been published.

**3. The Refresh click is the cadence — three streams cannot be recovered
later.** Everything CME publishes lives in a short rolling window and is
then gone from the free endpoints. Everything else backfills on demand.
That split is the single most important operational fact about this app:

| Recoverable at any time | Gone if you miss the window |
|---|---|
| COT (2y backfill in one call) | **Delivery** — current business date only |
| Spot (LBMA / EIA, full history) | **Futures bars** — rolling **5 business days** |
| Macro (DXY / VIX, 2y window) | **Options chains** — rolling **3 trade dates** |
| Inventory (8y from EIA) | |

The futures bar matters more than it used to: it now carries the high and
low that realized volatility is computed from, so a missed week is a
permanent hole in the volatility history, not just an absent settle price.
Within the window, gaps heal themselves automatically — the futures fetch
re-pulls every date CME offers and the options fetch asks for whichever
trade dates the database lacks.

**Practical rule: refresh at least twice a week, ideally daily.** Futures
forgive roughly a working week; options forgive about half of one.

This also gates the ML timeline: delivery-flow features join the models once
~60 delivery days have accrued (about three months), and the options family
joins once it covers enough of your selected date range — narrowing the date
range to the period you have chains for admits it much sooner.

### First-Run Checklist

After starting the app, walk through this once to confirm everything works
on your machine:

- [ ] **Refresh row**: five checkboxes (Delivery, Prices, Cot, Inventory,
      Options) and the Refresh button sit on their own line in the "Select"
      bar. Click Refresh with all five ticked → "⏳ Refreshing..." appears,
      and a few minutes later the status line reports each stream
      separately, e.g. `✓ Done — delivery ✓ (GC:1 ...) · prices ✓ · cot ✓ ·
      inventory ✓ · options ✓`. Without the EIA key the run still succeeds —
      CL/NG spot and all inventory are skipped with a log line.
- [ ] **GC → Delivery/COT, ~2-year range**: stat cards show live numbers
      (commercial net strongly negative for gold, latest COT date = last
      Tuesday); the COT, spot, volatility-regime, and correlation charts
      span the full range; the ML panels train (widen the range if they ask
      for more rows). The delivery charts show only the days fetched so far
      — they fill in daily. The delivery-vs-price scatter needs ~10
      delivery days before it appears.
- [ ] **CL/NG**: COT-focused view renders with the full weekly history;
      basis stays empty until the EIA key is set. **HG**: everything except
      basis (no copper spot yet).
- [ ] **Options view**: expiry dropdown shows near months ("Sep 2026");
      date chips switch days instantly; the strike-range chips
      (ATM ±10 / ±20 / ±50 / All, default ±10) resize the chain table and
      all charts together; IV shows "-" on deep-ITM strikes (correct — IV
      is undefined where the settle equals intrinsic value); smile, GEX,
      and Greeks panels have plausible shapes (gamma peaked near ATM, put
      deltas negative).
- [ ] **Cross-Asset, 2+ assets over ~2 years**: COT-ratio and basis
      overlays carry the view; the net-flow overlay and correlation heatmap
      stay thin until delivery history accrues.

---

## What This System Does

A multi-asset futures fundamentals dashboard that fuses four data streams —
each available to most market participants individually, but analytically
powerful together:

- **Delivery data** — daily firm-level physical delivery notices from
  CME/COMEX/NYMEX: which clearing firm issued or stopped how many contracts.
  Aggregated over time, this reveals which banks are accumulating or
  distributing physical supply.
- **COT positioning** — CFTC Commitment of Traders weekly data, kept at
  **full category detail**: producer/merchant, swap dealers, managed money
  and other reportables separately, rather than blended into "commercial vs
  speculator". That split is what distinguishes a miner hedging production
  from a bank hedging investor demand — two opposite pieces of news that the
  classic report reports as one number.
- **Volatility** — the asset's own realized volatility, measured from the
  futures bar's daily range, set against what the options market is charging.
  The gap between them says whether options are currently rich or cheap.
- **Prices & basis** — USD spot vs front-month futures. Basis (futures −
  spot) exposes contango/backwardation, storage economics, and near-term
  physical tightness. Includes vol-adjusted and carry-adjusted variants.
- **Options chains** — strike-level open interest, volume, settlement
  prices, and locally computed IV/Greeks: GEX (dealer gamma exposure), IV
  smile, put/call ratios, and OI migration day over day.
- **Inventories (crude and gas)** — weekly EIA stocks: US crude and Cushing
  (the NYMEX delivery point), SPR, gasoline, distillate and production; and
  natural gas working storage for the Lower 48 plus its five regions. Read
  against their own 5-year seasonal range, these are the primary fundamental
  driver for CL and NG the way delivery flow is for the metals.

**Everything is fetched automatically from free official sources** — CFTC's
public API, CME's settlement services and delivery reports, LBMA's official
prices, EIA's API, and Yahoo Finance for macro indicators. There is no
manual data entry anywhere in this app: click **Refresh**, and the data is
in the database and on the charts.

## Supported Assets

| Asset | Name | Delivery | Spot source |
|-------|------|----------|-------------|
| `GC` | Gold (COMEX) | ✓ | LBMA gold PM price |
| `SI` | Silver (COMEX) | ✓ | LBMA silver price |
| `HG` | Copper (COMEX) | ✓ | none free — basis unavailable until a source is chosen |
| `CL` | WTI Crude (NYMEX) | monthly notice period | EIA WTI Cushing (needs API key) |
| `NG` | Henry Hub Gas (NYMEX) | monthly notice period | EIA Henry Hub (needs API key) |

## Data Sources — what, from where, how often

Every row is a free, official, public source. Nothing is scraped, nothing is
paid, nothing is entered by hand.

| Stream | What arrives | Source | Source updates | Fetch this often | Cost of waiting longer |
|---|---|---|---|---|---|
| **Delivery** | Firm-level issued/stopped notices, MTD totals, settlement price | CME Issues & Stops PDFs (metals + energies) | Each business day, **current date only** | Daily on days you care about | **Permanent gap** |
| **Futures** | Front-month bar: month, OHLC, last, settle, volume, OI | CME settlement service | Each business day | **≥ 2×/week, ideally daily** | **Lost beyond 5 business days** |
| **Options** | Full chains: strike, side, settle, OHLC, volume, OI | CME options settlement service | Each business day, ~2-day lag | **≥ 2×/week, ideally daily** | **Lost beyond 3 trade dates** |
| **Spot** | Daily USD spot | LBMA JSON (GC/SI), EIA API (CL/NG) | Daily; EIA trails ~4 business days | Weekly is fine | Nothing — re-fetchable |
| **Macro** | USD Index, VIX | Yahoo chart API | Daily | Weekly is fine | Nothing — 2y window re-pulled |
| **COT** | Four CFTC trader categories, open interest | CFTC public API (`kh3c-gbw2`) | Weekly — Friday ~15:30 ET, for Tuesday | Weekly | Nothing — 2y backfill |
| **Inventory** | Weekly stocks/flows (CL, NG only) | EIA API | Weekly — Wed crude, Thu gas, for prior Friday | Weekly | Nothing — 8y backfill |
| **Risk-free rate** | 3-month T-bill, for option pricing | FRED CSV | Each business day | Automatic with Options | Nothing |

**The one rule worth memorising:** COT, spot, macro and inventory are
recoverable whenever you like. Delivery, futures bars and options chains are
not. That asymmetry is the entire reason to refresh regularly.

## How Each Source Is Fetched — and How Likely It Is to Break

Useful when something stops working, and for judging how much to depend on
each stream long-term.

- **CME — delivery, futures, options.** Fetched with `curl_cffi` using Chrome
  TLS impersonation, because CME's Akamai layer instantly rejects ordinary
  HTTP clients. **Moderate risk.** The options chain's request parameters were
  reverse-engineered from CME's own JavaScript, and the delivery PDF parser
  depends on where columns sit on the page — both could break on a site
  redesign. Failure is loud, not silent: you get a 403, zero rows, or a parse
  error named in the refresh status line. A failed request is retried once on
  a brand-new connection, because CME drops long-running ones.
- **CFTC — COT.** A documented public Socrata API returning plain JSON. **Low
  risk.** One known quirk: the field names are inconsistent (some carry `_all`
  suffixes, and `swap__positions_short_all` genuinely has a double
  underscore), so the code tries several spellings for each field. The
  realistic failure is a silent rename, which is why a sanity check rejects
  any week whose total positions exceed twice the open interest — a state real
  COT data cannot be in.
- **EIA — CL/NG spot and all inventory.** Official keyed API. **Low risk.**
  Needs `EIA_API_KEY`; without it these skip with an explicit log line rather
  than failing the refresh. Response times are erratic by nature (a fraction
  of a second warm, ~15 seconds cold, for the identical request).
- **LBMA — gold and silver spot.** A static JSON file of official prices.
  **Low risk.**
- **Yahoo — USD Index and VIX.** An undocumented public endpoint. **Highest
  risk in the set** — no contract, no deprecation notice, and its shape has
  changed before. If it breaks you lose the VIX regime chart and the
  vol-adjusted basis; nothing else is affected.
- **FRED — 3-month T-bill.** Keyless CSV download. **Low risk.**

## First-Time Setup

1. **Install Docker Desktop** ([docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/))
   and make sure it's running (icon in the menu bar / system tray).
2. **EIA API key** (for CL/NG spot — everything else works without it):
   register free at [eia.gov/opendata/register.php](https://www.eia.gov/opendata/register.php)
   (the key arrives by email), then copy `.env.example` in this folder to a
   new file named `.env` and paste the key after `EIA_API_KEY=` — no quotes,
   no spaces. `.env` stays on your machine; it is never committed. Until then,
   CL/NG spot is simply skipped and the log says exactly that.
3. **Start the app**: double-click `start.command` (Mac) or `start.bat`
   (Windows). The first build takes a few minutes; afterwards starts are
   quick (Docker only rebuilds what changed). The browser opens to
   `http://localhost:8057` when the dashboard is actually up — if the
   container fails or the dashboard doesn't respond within 60 seconds, the
   script says so instead of claiming success.
4. **Stop**: double-click `stop.command` / `stop.bat`.

Running without Docker instead: `pip install -e ".[dev]"` (or
`pip install -r requirements.lock`), then `python3 app.py`.

## Daily Use — the Refresh Row

Inside the "Select" bar, on its own line, are five checkboxes and a
**Refresh** button:

- **Delivery** — today's CME Issues & Stops reports (metals + energies).
  The PDFs exist **only for the current business date** — fetch every
  trading day you care about; a skipped day is a permanent gap in
  firm-level history (see "How history accrues" below). The energy report
  being empty outside the monthly delivery notice period is normal.
- **Prices** — LBMA/EIA spot, the CME front-month **bar** (contract month,
  open/high/low/last/settle, volume and open interest) for the ~5 most recent
  trade dates, plus USD Index and VIX. Spot and macro backfill ~2 years on an
  empty database. The futures bar does **not** backfill: CME serves only that
  5-day window, and the high and low are what realized volatility is built
  from, so a missed week is a permanent hole in the vol history. Gaps inside
  the window heal automatically — every offered date is re-fetched each time.
- **Cot** — CFTC weekly positioning. Backfills its entire gap automatically
  (~2 years on a fresh database); fetching once a week is plenty. New
  reports publish Friday afternoons for Tuesday's positions.
- **Inventory** — weekly EIA stocks for CL and NG (the metals have none, and
  are skipped silently). Eight years of history on a fresh database, so the
  5-year seasonal bands are properly filled. Crude numbers come from the
  Wednesday Weekly Petroleum Status Report, gas from the Thursday storage
  report, both describing the prior Friday week-ending date — so a fresh
  reading is already several days old. Weekly fetching is plenty; gaps
  backfill themselves.
- **Options** — full settlement chains (strike-level OI/volume/settle) for
  every monthly expiry within the next 3 months, for each recent trade date
  not already stored. IV and Greeks are computed locally from the stored
  prices — no vendor feed.

Tick any subset, click Refresh. The status line reports each stream
separately (`✓ Done — delivery ✓ (GC:1 SI:1 ...) · cot ✓ ...`), and every
view updates automatically the moment its underlying data changed — there
is no separate "recompute" step.

### How long a refresh takes

Every stream fetches only what it's missing, so a routine refresh is far
cheaper than the first one. Rough shape of the wait, slowest first:

- **Options** is the big one — it asks CME for a chain per expiry per trade
  date, so it dominates a full refresh. Untick it if you only want today's
  fundamentals.
- **Inventory** is 12 small requests, but EIA's response time is erratic —
  typically a fraction of a second, occasionally ~15 seconds for the same
  request. A cold run can therefore take a couple of minutes and a warm one
  under ten seconds, for identical data. Nothing is wrong when this happens;
  a request that exceeds the client's 45-second limit is retried
  automatically (you'll see one warning line in the container log).
- **Prices** is a handful of requests per asset; **COT** is one per asset;
  **Delivery** is two PDFs.

If a refresh feels slow, it is almost always Options or a cold EIA moment —
not a hang.

### How history accrues

Spot, macro, COT and inventory backfill themselves. **The three CME streams
cannot** — delivery detail exists only for the current business date,
futures bars only for the last 5 business days, and options chains only for
the last 3 trade dates. Those three histories start the day you start
refreshing and grow from there.

The ML charts work from day one regardless: they train on whatever families
have enough history and enough coverage (price, COT, volatility, and for
CL/NG inventory), and the newer families join automatically as they accrue —
delivery flow after roughly 60 delivery days, realized volatility after 20
futures bars, and the options family once it covers enough of your selected
range. This is a property of the sources, not a bug.

## The Dashboard

One page, driven by the collapsible "Select" bar:

- **Mode** — `Single Asset` or `Cross-Asset`.
- **Asset** — one asset at a time (Single Asset mode).
- **Date range** — defaults to the last 30 days; widen for more history and
  for the ML charts' 60-row minimum.
- **View** — `Delivery/COT` or `Options` (greyed out when the asset has no
  chains stored).
- **Options Expiry** — shown in Options view; e.g. "Sep 2026".

### Delivery/COT view

- **Delivery Alert banner (red)** — today's activity is a statistical
  outlier (|z| > 2 on net flow or stopped). "Worth a closer look", not
  automatically bullish or bearish.
- **Stat cards** — commercial/speculator net (with week-over-week %),
  open interest, latest COT date, issued/stopped/MTD/net flow.
- **Daily Delivery Volumes / Net Flow / MTD** — issued vs stopped per day;
  net flow (stopped − issued) is the single most useful daily number:
  positive = physical being taken (demand), negative = supply pressure.
- **Top Firms / Firm Behavior** — who's active, split House ('H', the
  firm's own book) vs Customer ('C', client flow) — these are distinct
  economic actors and never merged. Click a firm chip to see one firm's
  history; the chart title names the firm and account type.
- **Correlation heatmap** — pairwise correlations across the engineered
  features.
- **Net Flow vs Future Returns** — does delivery pressure lead price? Each
  point is a day; the dashed line is the fitted trend.
- **ML panels (4 charts)** — regression RMSE/R², classification accuracy
  (50% = coin flip), feature importance, predicted-vs-actual. These grade
  whether the feature set carries signal — a screening aid, not a trading
  signal. They appear once the range holds ≥ 60 aligned rows (widen the date
  range if they show the "needs more rows" message). The inputs are not
  fixed: each family joins once it has enough history of its own, so the
  models start on price/COT/volatility, gain inventory for CL and NG, and
  gain delivery flow after a few months of daily fetching.

  A feature also has to cover enough of the range, not merely exist. The
  models train on dates where **every** selected input is present, so
  admitting one thin family would shrink the training set for all of them —
  a hundred rows with forty inputs is worse than five hundred rows with
  thirty. Features that would cause that are listed as *coverage-rejected*
  rather than used. If the options family is sitting in that list, the fix
  is to **narrow the date range** to the period you actually have chains
  for: within that window its coverage is high and it joins immediately.
  **Feature
  Importance is the panel that tells you which inputs are actually carrying
  the model** — on crude, the inventory deviation and weekly build/draw
  ranked top immediately. Expect small or negative R²: these are noisy
  forward returns, and a negative value means "worse than guessing the
  average", which is information too.
- **Basis Analysis** — basis over time and basis vs delivery activity; the
  title carries the convenience-yield model's R² when computable. Positive
  basis = contango (normal, storage costs); negative = backwardation
  (near-term physical tightness).
- **COT charts** — positioning vs delivery flow; extreme-positioning
  z-score with ±2σ markers (the contrarian signal); commercial vs
  speculator nets and their spread.
- **Producer vs Swap Dealer** — the split the word "commercial" hides, and
  often the most informative chart on the page. Both categories are lumped
  together as "commercial" in the classic report, but they mean opposite
  things. **Producer/merchant** is physical: miners and drillers selling
  future output forward. When producer shorts build into a rally, that is
  real supply arriving at a known price, and it tends to cap the move.
  **Swap dealers** are intermediaries — usually the mirror image of investor
  demand, since a bank hedges client and ETF-related exposure in futures. A
  growing swap-dealer short therefore often means *investors are buying*: a
  bullish flow that shows up in the blended number as a bearish short. Red
  ✕ markers flag weeks where the two moved in opposite directions and
  largely cancelled — weeks where the blended commercial line reads "nothing
  happened" while a great deal happened underneath it. The dotted line is
  **managed money**, the speculative-positioning figure that commodity desks
  actually quote (previously diluted here with other reportables).
- **Volatility Regime** — VIX with calm/elevated bands and the current
  low/normal/high classification. Note what this is: the **S&P 500's**
  expected volatility, i.e. a global risk-appetite proxy. It is not a measure
  of your commodity's own volatility — that is the next chart.
- **Volatility Risk Premium** — the asset's *own* volatility, and what
  options charge for it. The top panel plots implied volatility (what the
  options market is pricing, at a constant 30-day horizon) against realized
  volatility (how much the front-month future has actually been moving over
  the last 20 sessions). The lower panel is the gap between them.
  **Positive means options are rich** — you are paying a premium over what
  the market has been delivering, so spreads or premium-selling structures
  express a view more efficiently than buying outright. **Negative or
  compressed means options are cheap** relative to actual movement, which
  favours buying optionality outright, and often precedes a volatility
  expansion. This is the only reading in the app that speaks to *how* to
  take a position rather than *whether* to. Natural gas normally carries the
  largest and most violent premium (winter weather); crude's spikes into
  OPEC meetings and decays afterwards; gold's is comparatively stable, which
  makes its excursions more meaningful.
- **Seasonality / Firm Concentration (HHI)** — monthly seasonal patterns;
  daily delivery concentration (0.25 = moderate, 0.50 = high — a few firms
  dominating the delivery flow).

**Inventory section (CL and NG only)** — sits directly under the stat cards,
above the positioning charts it helps explain. Chips switch between the
series (US Crude, Cushing, SPR, Gasoline, Distillate, Production; or Lower 48
and its five regions). Switching a series redraws only this section and its
stat cards — the rest of the page stays put, including your scroll position —
and your choice sticks when you move between assets, falling back to the
first series if the one you picked doesn't exist for that asset:

- **vs 5-Year Seasonal Range** — the current level over a shaded band of the
  same week's min–max across the previous five years, with that window's
  average dashed. Inventories are strongly seasonal — gas injects all summer
  and withdraws all winter — so the raw level says little; where it sits
  *relative to its own seasonal norm* is the signal. The earliest years of
  history have no band (they have no prior years to compare against), which
  is expected rather than missing data.
- **Weekly Build / Draw** — the change that moves the market on report day.
  Bars follow the sign of the change, and the title states the direction
  plainly: a build adds supply and is bearish for price; a draw tightens and
  is bullish. Don't read the colors as a price call.
- **Surplus / Deficit vs 5-Year Average** — the tight-or-loose read over
  time. A persistent deficit (Cushing well below its seasonal norm, say)
  is the classic backwardation setup; a persistent surplus is the opposite.

Assets with COT but no delivery history (CL/NG most of the time) show the
COT-focused chart subset automatically — the inventory section appears in
both cases.

### Options view

- **Date chips** — one per stored trade date; click to jump between days.
- **Strike range chips** — `ATM ±10 / ±20 / ±50 / All`, default **±10**.
  CME lists around 570 strikes for a gold month, so the full chain is a wall
  of numbers; ±10 shows the ten listed strikes immediately below the
  underlying price and the ten immediately above it. "ATM" here is the
  underlying price row itself, not a strike, so both sides always get the
  same count. The range applies to the table **and** all six charts
  together — which also means the underlying row's totals, PCR, OI
  difference and the GEX chart describe the strikes you're looking at, not
  the whole chain. Widen to ±50 or All when you want the wings (deep OI
  clusters and far-strike gamma often sit out there). Changing the range
  redraws the table and charts only — the date chips and your scroll position
  stay where they are.
- **Chain table** — calls left, puts right, strikes centered; the underlying
  price row (brass) carries totals, PCR, OI difference, and IV skew.
  Color-coding: ITM vs OTM shading, top-2 OI strikes highlighted, OI
  changes colored with the day's largest build/unwind bordered.
- **OI / Volume / OI-change distributions** — where positioning sits and
  where it moved since the previous stored date. Large OI clusters near the
  money often act as informal support/resistance.
- **IV Smile** — implied volatility by strike per side. One side
  consistently richer = the market paying up for protection there.
- **GEX** — dealer gamma exposure by strike. Positive zones tend to dampen
  moves (dealers fade); negative zones amplify them (dealers chase).
- **Greeks panel** — delta/gamma/theta/vega by strike, calls vs puts.
- **Options History** (below the chain, unaffected by the strike chips) —
  three series showing how the surface has *moved*, which the chain alone
  cannot tell you:
  - **ATM implied volatility**, at a constant 30-day horizon. The constant
    horizon matters: a given expiry's remaining life shrinks from ~90 days
    to 1, and implied vol depends on time to expiry, so plotting "the front
    expiry's IV" would sawtooth every month by roughly the size of the
    signal. This interpolates between expiries to a fixed 30 days instead.
  - **Put/Call ratio** by open interest, with 1.0 marked. Above 1 means more
    open put interest than call — defensive positioning; below, the reverse.
  - **25-delta skew** — the price of downside protection minus the price of
    upside, compared at equal and opposite delta rather than at fixed
    strikes (CME's strike spacing is uneven, so a strike-based comparison
    silently changes meaning across the surface). Positive is the normal
    state for most commodities; a sharp move toward *negative* — calls
    richer than puts — is a classic supply-shock tell in crude.

IV shows "-" on deep-in-the-money strikes whose settle equals intrinsic
value — IV is genuinely undefined there, not missing.

### Cross-Asset mode

Pick 2+ assets: z-score-normalised net-flow overlay, commercial net ratio
overlay, basis % overlay, and a pairwise net-flow correlation heatmap over
the last 60 rows. Defaults to a 2-year range.

## Practical Use — a Working Routine

### The refresh habit

**Daily, or at least twice a week** — tick everything and click Refresh.
The three CME streams (Delivery, Prices, Options) are the ones with no
second chance; the rest could be fetched monthly and lose nothing. If you
are in a hurry, Delivery + Prices takes seconds; Options is the slow one.

**Weekly is enough for COT and Inventory** — they publish weekly and
backfill completely, so there is no such thing as missing them.

### Which view answers which question

| Question | Where to look |
|---|---|
| Is physical metal actually moving? | Delivery/COT → net flow, top firms, HHI |
| Who is on the other side, and do they mean it? | Producer vs Swap Dealer |
| Are speculators crowded? | Managed money net, COT extremes z-score |
| Is the market tight or loose? | Inventory deviation vs 5-year average (CL/NG); basis / backwardation (all) |
| Are options expensive right now? | Volatility Risk Premium |
| Where will hedging flows pin price? | Options → GEX, top OI strikes |
| What is the market afraid of? | Options History → 25-delta skew, term structure |
| Does any of this predict returns? | ML panels — as a screening aid, never a signal |

### Reading a metal vs reading an energy

**Metals (GC, SI, HG)** — delivery is the fundamental. Watch net flow and
firm concentration, read them against COT positioning, and use the options
view for how the market is priced. Copper has no spot source, so its basis
and spot-return charts stay empty by design (LME's cash price is licensed);
everything else works.

**Energies (CL, NG)** — inventory is the fundamental, and delivery notices
only exist during the monthly notice period. Most of the year these show the
COT-focused layout with the inventory section: read the level against its
five-year band first, then the weekly build/draw, then positioning. Natural
gas is where the volatility premium is most dramatic.

### A five-minute pass

1. Refresh with everything ticked.
2. Delivery/COT, ~2 years. Check the alert banner and the stat cards.
3. Producer vs Swap Dealer — is commercial positioning one story or two?
4. For CL/NG, the inventory deviation chart — tight or loose?
5. Volatility Risk Premium — if you are considering options, is this a
   moment to buy them or to sell them?
6. Options view for the strikes that matter, then the history charts for
   whether today is unusual.

### Things that look broken but are not

- HG basis charts empty → no free copper spot exists.
- CL/NG basis empty → EIA key not set, or EIA's ~4-day publication lag.
- IV showing "-" on deep in-the-money strikes → IV is genuinely undefined
  where an option settles at intrinsic value.
- Realized volatility and the options charts empty on a fresh database →
  they accrue from your first refresh; there is no history to backfill.
- ML panels listing features as coverage-rejected → narrow the date range.
- Energy delivery report empty outside the notice period → normal.

## Where the Data Lives

`data/fundamentals.db` — a single SQLite file, created automatically on
first start, holding only raw observed data (every derived number is
computed on demand). It persists across container rebuilds via the Docker
volume. **Deleting it is not free.** The next Refresh re-backfills spot,
macro, COT and inventory in full — but past delivery days, futures bars
older than 5 business days, and option chains older than 3 trade dates are
gone permanently, because the free sources no longer serve them.

## Security Notes

The app assumes nothing untrusted runs on your machine and is reachable
only from it: Docker publishes the port to `127.0.0.1` only (invisible to
your LAN and the internet), and the app sends defense-in-depth headers
(CSP, frame denial, no-referrer, fixed-origin CORS). There is deliberately
no login screen — the loopback-only binding is the access control. If you
ever want to reach the dashboard from another device, revisit this section
first rather than just widening the port binding.

## Licence

MIT — see [LICENSE](LICENSE).

The bundled IBM Plex typefaces are licensed separately under the SIL Open Font
License 1.1 — see [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).

From [ChorBazaar](https://chorbazaar.co.in).
