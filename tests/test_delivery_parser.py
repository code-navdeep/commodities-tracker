"""Delivery section parser — fixture is a realistic fixed-width GC section."""

from pathlib import Path

from commodities.sources.delivery import parse_section

FIXTURE = (Path(__file__).parent / 'fixtures' / 'delivery_section.txt').read_text()


def test_parses_all_header_fields():
    notice = parse_section('GC', FIXTURE)
    assert notice is not None
    assert notice.contract_month == '2026-08'
    assert notice.intent_date == '2026-08-07'
    assert notice.delivery_date == '2026-08-11'
    assert notice.settlement == 4340.7
    assert notice.total_issued == 286
    assert notice.total_stopped == 286
    assert notice.month_to_date == 15416


def test_firm_sums_reconcile_to_totals():
    notice = parse_section('GC', FIXTURE)
    assert sum(f.issued for f in notice.firms) == notice.total_issued
    assert sum(f.stopped for f in notice.firms) == notice.total_stopped


def test_single_number_lines_classified_by_column():
    notice = parse_section('GC', FIXTURE)
    by_code = {(f.firm_code, f.org_code): f for f in notice.firms}
    assert by_code[('118', 'H')].stopped == 2 and by_code[('118', 'H')].issued == 0
    assert by_code[('435', 'H')].issued == 160 and by_code[('435', 'H')].stopped == 0
    assert by_code[('555', 'C')].issued == 115
    assert by_code[('661', 'C')].stopped == 265


def test_both_number_line_and_awkward_names():
    notice = parse_section('GC', FIXTURE)
    by_code = {(f.firm_code, f.org_code): f for f in notice.firms}
    dorman = by_code[('152', 'C')]
    assert dorman.firm_name == 'DORMAN TRADING, LLC'      # comma in name
    assert (dorman.issued, dorman.stopped) == (9, 11)     # two-number row
    assert by_code[('435', 'H')].firm_name == 'SCOTIA CAPITAL (USA)'  # parens


def test_missing_required_fields_fails():
    assert parse_section('GC', "SOME RANDOM TEXT\nWITH NO STRUCTURE") is None
    no_firms = ("CONTRACT: AUGUST 2026 COMEX 100 GOLD FUTURES\n"
                "INTENT DATE: 08/07/2026  DELIVERY DATE: 08/11/2026\n"
                "TOTAL: 0 0")
    assert parse_section('GC', no_firms) is None
