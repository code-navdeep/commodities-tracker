"""Delivery source — CME Daily Issues & Stops PDFs.

Two PDFs (metals DLV615-T, energies DLV602-T), published only for the current
business date. Text is extracted with layout preserved — character column
positions are what classify single-number firm lines as issued vs stopped.
An empty energy report ("NO DATA FOUND") outside the monthly notice period is
normal and yields zero notices.

parse_section() is a pure text → DeliveryNotice function, content-driven:
the contract line (month name + year), the INTENT/DELIVERY date line, the
SETTLEMENT line, and the ISSUED/STOPPED header are recognised by what they
contain, wherever they appear — page banners and EXCHANGE: lines are simply
never matched.
"""

import io
import logging
import re
from datetime import datetime

import pdfplumber

from ..config import ASSETS
from ..domain import DeliveryNotice, FirmLine
from .http import get_bytes

logger = logging.getLogger(__name__)

PDF_URLS = [
    'https://www.cmegroup.com/delivery_reports/MetalsIssuesAndStopsReport.pdf',
    'https://www.cmegroup.com/delivery_reports/EnergiesIssuesAndStopsReport.pdf',
]

MONTHS = {
    'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4, 'MAY': 5, 'JUNE': 6,
    'JULY': 7, 'AUGUST': 8, 'SEPTEMBER': 9, 'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12,
}

# Firm line: code, org letter, name (incl. parens/commas — 'SCOTIA CAPITAL
# (USA)', 'DORMAN TRADING, LLC'), then optionally one or two numbers.
FIRM_LINE_RE = re.compile(r'^(\d+)\s+([A-Z])\s+([A-Z\s&\.\-(),]+?)(?:\s+([\d,]+))?(?:\s+([\d,]+))?\s*$')


def _to_int(text: str) -> int:
    return int(text.replace(',', ''))


def _parse_contract_month(line: str) -> str | None:
    """'CONTRACT: AUGUST 2026 COMEX 100 GOLD FUTURES' -> '2026-08'. Only a
    real month name followed by a year qualifies — which is what separates a
    contract line from every other line in the report."""
    match = re.match(r'\s*(?:CONTRACT:\s*)?([A-Z]+)\s+(\d{4})\b', line.strip(), re.IGNORECASE)
    if not match:
        return None
    month = MONTHS.get(match.group(1).upper())
    if not month:
        return None
    return f"{match.group(2)}-{month:02d}"


def _parse_dates(line: str) -> tuple[str, str] | None:
    match = re.search(
        r'INTENT DATE:\s*(\d{1,2}/\d{1,2}/\d{4}).*DELIVERY DATE:\s*(\d{1,2}/\d{1,2}/\d{4})',
        line, re.IGNORECASE)
    if not match:
        return None
    return tuple(datetime.strptime(g, '%m/%d/%Y').strftime('%Y-%m-%d') for g in match.groups())


def _parse_settlement(line: str) -> float | None:
    match = re.search(r'SETTLEMENT:\s*([\d,]+\.?\d*)', line, re.IGNORECASE)
    if not match:
        return None
    try:
        value = float(match.group(1).replace(',', ''))
        return value if value > 0 else None
    except ValueError:
        return None


def _parse_firm_line(line: str, issued_col, stopped_col) -> FirmLine | None:
    stripped = line.strip()
    if not stripped or 'FIRM' in stripped or 'TOTAL' in stripped or 'MONTH TO DATE' in stripped:
        return None
    match = FIRM_LINE_RE.match(stripped)
    if not match:
        return None

    firm_code, org_code, firm_name, num1, num2 = match.groups()
    issued = stopped = 0

    if num1 and num2:
        issued, stopped = _to_int(num1), _to_int(num2)
    elif num1:
        # Single number: classify by which header column it sits under —
        # distance of its character position to ISSUED vs STOPPED.
        value = _to_int(num1)
        num_match = re.search(r'[\d,]+\s*$', line.rstrip())
        num_pos = num_match.start() if num_match else -1
        if num_pos >= 0 and issued_col is not None and stopped_col is not None:
            if abs(num_pos - issued_col) <= abs(num_pos - stopped_col):
                issued = value
            else:
                stopped = value
        else:
            stopped = value   # no header info — stopped is the common case

    return FirmLine(firm_code, org_code, firm_name.strip(), issued, stopped)


def parse_section(asset: str, text: str) -> DeliveryNotice | None:
    """Parse one CONTRACT section. None when required fields are missing;
    firm-sum vs TOTAL mismatches warn but do not fail."""
    contract_month = None
    intent_date = delivery_date = None
    settlement = None
    total_issued = total_stopped = month_to_date = 0
    issued_col = stopped_col = None
    firms = []

    for line in text.split('\n'):
        upper = line.upper()

        if 'TOTAL:' in upper:
            match = re.search(r'TOTAL:\s+([\d,]+)\s+([\d,]+)', line)
            if match:
                total_issued, total_stopped = _to_int(match.group(1)), _to_int(match.group(2))
            continue
        if 'MONTH TO DATE' in upper:
            match = re.search(r'MONTH TO DATE:\s+([\d,]+)', line, re.IGNORECASE)
            if match:
                month_to_date = _to_int(match.group(1))
            continue
        if 'INTENT DATE' in upper and intent_date is None:
            dates = _parse_dates(line)
            if dates:
                intent_date, delivery_date = dates
            continue
        if 'SETTLEMENT' in upper and settlement is None:
            settlement = _parse_settlement(line)
            continue
        if 'ISSUED' in upper and 'STOPPED' in upper and issued_col is None:
            issued_col, stopped_col = upper.find('ISSUED'), upper.find('STOPPED')
            continue
        if contract_month is None:
            contract_month = _parse_contract_month(line)
            if contract_month:
                continue
        firm = _parse_firm_line(line, issued_col, stopped_col)
        if firm:
            firms.append(firm)

    if not (contract_month and intent_date and delivery_date and firms):
        logger.error(f"{asset}: delivery section missing required fields "
                     f"(month={contract_month}, intent={intent_date}, firms={len(firms)})")
        return None

    sum_issued = sum(f.issued for f in firms)
    sum_stopped = sum(f.stopped for f in firms)
    if sum_issued != total_issued:
        logger.warning(f"{asset}: firm issued sum {sum_issued} != TOTAL {total_issued}")
    if sum_stopped != total_stopped:
        logger.warning(f"{asset}: firm stopped sum {sum_stopped} != TOTAL {total_stopped}")

    return DeliveryNotice(
        asset=asset, contract_month=contract_month,
        intent_date=intent_date, delivery_date=delivery_date,
        total_issued=total_issued, total_stopped=total_stopped,
        month_to_date=month_to_date, settlement=settlement,
        firms=tuple(firms))


def _extract_text(pdf_bytes: bytes) -> str:
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text(layout=True) or '')
    return '\n'.join(pages)


def _split_sections(text: str) -> list[str]:
    lines = text.split('\n')
    starts = [i for i, line in enumerate(lines) if re.match(r'\s*CONTRACT:', line)]
    return ['\n'.join(lines[start:(starts[n + 1] if n + 1 < len(starts) else len(lines))])
            for n, start in enumerate(starts)]


def _asset_for(section: str) -> str | None:
    contract_line = section.split('\n', 1)[0].upper()
    for asset, spec in ASSETS.items():
        if any(keyword in contract_line for keyword in spec['delivery_keywords']):
            return asset
    return None


def fetch_delivery() -> list[DeliveryNotice]:
    """Today's notices for every registry asset across both PDFs."""
    notices = []
    for url in PDF_URLS:
        pdf_bytes = get_bytes(url)
        sections = _split_sections(_extract_text(pdf_bytes))
        logger.info(f"{url.rsplit('/', 1)[-1]}: {len(sections)} contract section(s)")
        for section in sections:
            asset = _asset_for(section)
            if asset is None:
                continue
            notice = parse_section(asset, section)
            if notice:
                notices.append(notice)
    return notices
