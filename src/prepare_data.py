"""Extract the EU27 non-resident overnight-stay panel from the downloaded Eurostat workbook.

Dataset: Eurostat tour_occ_nim, extracted 2026-10-23.
This script uses the already downloaded workbook; it does not access the network.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "tour_occ_nim__custom_23540219_spreadsheet.xlsx"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

EU27 = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE",
    "EL", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT",
    "RO", "SK", "SI", "ES", "SE",
}
TARGET_CODES = {
    "I551-I553": "overnight_stays_total",
    "I551": "overnight_stays_hotels",
    "I552_I553": "overnight_stays_other_accommodation",
}


def to_numeric(value):
    if value is None or (isinstance(value, str) and value.strip() in {"", ":"}):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Expected downloaded Eurostat workbook at {RAW}")

    wb = load_workbook(RAW, read_only=True, data_only=True)
    records = []
    selected = []
    for ws in wb.worksheets[:39]:  # first unit block: Number [NR]
        # The workbook's XML dimension is not always populated; explicit bounds are used.
        rows = list(ws.iter_rows(max_row=27, max_col=644, values_only=True))
        unit_label = str(rows[5][2])
        geo_label = str(rows[6][2])
        match = re.search(r"\[([A-Z]{2})\]", geo_label)
        if "[NR]" not in unit_label or not match:
            continue
        geo = match.group(1)
        if geo not in EU27:
            continue

        dates = []
        for col, value in enumerate(rows[8]):
            if col >= 4 and isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}", value):
                dates.append((value, col))

        by_code = {}
        for row in rows[10:20]:
            if row[0] == "FOR" and row[2] in TARGET_CODES:
                by_code[row[2]] = row

        country_name = geo_label.rsplit(" [", 1)[0]
        selected.append((geo, country_name, ws.title))
        for month, col in dates:
            item = {"geo_code": geo, "country": country_name, "month": month}
            for code, column_name in TARGET_CODES.items():
                row = by_code.get(code)
                item[column_name] = to_numeric(row[col]) if row is not None else np.nan
                # The next column is the Eurostat observation flag; retain it for audit.
                item[column_name + "_flag"] = row[col + 1] if row is not None and col + 1 < len(row) else None
            records.append(item)

    wb.close()
    data = pd.DataFrame(records)
    data["month"] = pd.PeriodIndex(data["month"], freq="M")
    data = data.sort_values(["geo_code", "month"]).reset_index(drop=True)
    data.to_csv(OUT / "eurostat_eu27_nonresident_all_months.csv", index=False)

    sample = data[(data["month"] >= pd.Period("2015-01", "M")) & (data["month"] <= pd.Period("2025-12", "M"))].copy()
    sample.to_csv(OUT / "eurostat_eu27_nonresident_2015_2025.csv", index=False)

    summary = []
    for col in TARGET_CODES.values():
        for period, frame in [("2015-2025", sample), ("2015-2021", sample[sample.month <= pd.Period("2021-12", "M")]),
                              ("2022", sample[sample.month.dt.year == 2022]), ("2023", sample[sample.month.dt.year == 2023]),
                              ("2024-2025", sample[sample.month >= pd.Period("2024-01", "M")])]:
            summary.append({"series": col, "period": period, "expected_rows": len(frame),
                            "observed": int(frame[col].notna().sum()), "missing": int(frame[col].isna().sum()),
                            "missing_rate": float(frame[col].isna().mean()) if len(frame) else np.nan})
    pd.DataFrame(summary).to_csv(OUT / "data_quality_summary.csv", index=False)

    meta = [
        "Dataset: Eurostat tour_occ_nim",
        "Workbook: tour_occ_nim__custom_23540219_spreadsheet.xlsx",
        "Workbook extracted: 2026-10-23; Eurostat data update shown in workbook: 2026-10-22",
        "Selection: monthly frequency, unit Number (NR), country of residence FOR, EU27 member states",
        "Target: NACE I551-I553 (hotels; holiday and other short-stay accommodation; camping grounds, RV parks and trailer parks)",
        "Auxiliary lagged subseries: I551 (hotels) and I552_I553 (holiday and other short-stay accommodation; camping grounds etc.)",
        f"EU27 countries extracted: {len(selected)}",
        f"Long panel rows: {len(data)}; date range: {data.month.min()} to {data.month.max()}",
        "Note: auxiliary accommodation categories are subseries of the same Eurostat dataset, not independent external data sources.",
        "Missing markers ':' are preserved as NaN; no zero-filling is performed.",
    ]
    (OUT / "data_provenance.txt").write_text("\n".join(meta) + "\n", encoding="utf-8")
    print("Extracted countries:", len(selected))
    print("Long rows:", len(data), "range:", data.month.min(), "to", data.month.max())
    print("Sample rows:", len(sample))
    print(pd.read_csv(OUT / "data_quality_summary.csv").to_string(index=False))
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
