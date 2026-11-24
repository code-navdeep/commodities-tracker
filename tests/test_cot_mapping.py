"""COT field mapping — fixture is the real gold 2025-09-23 API record, whose
mapped values were verified identical to CFTC's published HTM report."""

import json
from pathlib import Path

from commodities.domain import CotWeek
from commodities.sources.cot import map_record, validate

RECORD = json.loads((Path(__file__).parent / 'fixtures' / 'cftc_gold_20250923.json').read_text())


def _week(**overrides):
    fields = dict(prod_long=0, prod_short=0,
                  swap_long=0, swap_short=0, swap_spreading=0,
                  mm_long=0, mm_short=0, mm_spreading=0,
                  other_long=0, other_short=0, other_spreading=0,
                  nonreportable_long=0, nonreportable_short=0, open_interest=100)
    fields.update(overrides)
    return CotWeek('GC', '2026-01-01', **fields)


def test_categories_are_stored_as_published():
    """Nothing is summed in the source — each reportable category survives
    intact, which is what makes producer-vs-swap-dealer readable."""
    week = map_record(RECORD, 'GC')
    assert week.report_date == '2025-09-23'
    assert (week.prod_long, week.prod_short) == (38666, 95508)
    assert (week.swap_long, week.swap_short, week.swap_spreading) == (42028, 276247, 97855)
    assert (week.mm_long, week.mm_short, week.mm_spreading) == (192350, 33734, 89092)
    assert (week.other_long, week.other_short, week.other_spreading) == (127422, 26778, 242301)
    assert week.nonreportable_long == 68734
    assert week.nonreportable_short == 36933
    assert week.open_interest == 898447


def test_legacy_aggregates_still_reproduce_the_published_report():
    """The commercial / non-commercial groupings moved to features.py; they
    must still sum to exactly what CFTC publishes."""
    week = map_record(RECORD, 'GC')
    assert week.prod_long + week.swap_long == 80694
    assert week.prod_short + week.swap_short == 371755
    assert week.mm_long + week.other_long == 319772
    assert week.mm_short + week.other_short == 60512
    assert week.mm_spreading + week.other_spreading == 331393


def test_valid_week_passes_guard():
    assert validate(map_record(RECORD, 'GC')) is None


def test_guard_rejects_positions_inconsistent_with_oi():
    week = _week(prod_long=500, mm_long=400, nonreportable_long=300, open_interest=100)
    assert 'inconsistent' in validate(week)


def test_guard_rejects_negative_values():
    assert validate(_week(prod_long=-1)) == 'negative position value'


def test_guard_sees_every_category():
    """The tripwire exists to catch an upstream field rename, so a bad value
    in any category — not just the first — must trip it."""
    assert validate(_week(other_spreading=-5)) == 'negative position value'
