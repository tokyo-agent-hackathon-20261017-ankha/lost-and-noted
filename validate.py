"""Validation framework: check stored records against a golden expectation file.

Usage:
    .venv/bin/python validate.py expectations.json [--data-dir data]

Expectation file format (JSON):
{
  "20261017-100000_100": {            # key = image filename stem (DATETIME_ROOM)
    "must_contain": [                 # item types that MUST be detected (substring match)
      {"type": "key card", "class": "company_property"}
    ],
    "must_not_contain": [             # item types that must NOT appear
      {"type": "lighter", "class": "flammable"}
    ],
    "min_items": 1,
    "max_items": 10,
    "room_number": 100
  }
}

Each must_contain entry: {"type": <substring, case-insensitive, optional>,
                          "class": <exact class, optional>}
At least one of type/class must be given. Exit code 0 = all pass, 1 = failures.
"""
import argparse
import json
import sys
from pathlib import Path

VALID_CLASSES = {
    "expirable", "non_expirable", "flammable", "company_property",
    "domestic_currency", "foreign_currency", "valuable",
}
VALID_STATUSES = {"in_storage", "in_safe", "to_policebox"}  # statuses are free-form, these are the defaults


def load_records(data_dir: Path) -> list[dict]:
    """Load all records from the records directory."""
    return [
        json.loads(p.read_text())
        for p in sorted((data_dir / "records").glob("*.json"))
    ]


def find_records(records: list[dict], key: str, exp: dict) -> list[dict]:
    """Find records matching an expectation key.

    Key may be:
      - a stored image stem prefix, e.g. '20261017-030205_100'
      - an asset filename stem, e.g. '20261017_100' or 'company-property-room-card'
        (matched via room_number, if given in the expectation or trailing '_<room>')
    """
    hits = [r for r in records
            if any(Path(img).stem.startswith(key) for img in r.get("images", []))]
    if hits:
        return hits
    # derive room number: explicit field, or trailing _<digits> in the key
    room = exp.get("room_number")
    if room is None and "_" in key:
        tail = key.rsplit("_", 1)[1]
        if tail.isdigit():
            room = int(tail)
    if room is not None:
        return [r for r in records if r.get("room_number") == room]
    return []


def match_item(item: dict, spec: dict) -> bool:
    type_ok = "type" not in spec or spec["type"].lower() in item.get("type", "").lower()
    class_ok = "class" not in spec or spec["class"] == item.get("class")
    return type_ok and class_ok


def check_record(rec: dict, exp: dict) -> list[str]:
    errors = []
    items = rec.get("items", [])

    for i, item in enumerate(items):
        if item.get("class") not in VALID_CLASSES:
            errors.append(f"item[{i}] invalid class: {item.get('class')!r}")
        if item.get("status") not in VALID_STATUSES:
            errors.append(f"item[{i}] non-default status: {item.get('status')!r}")
        if not item.get("type"):
            errors.append(f"item[{i}] missing type")
        for field in ("id", "created", "last_updated"):
            if not item.get(field):
                errors.append(f"item[{i}] missing {field}")

    if "min_items" in exp and len(items) < exp["min_items"]:
        errors.append(f"expected >= {exp['min_items']} items, got {len(items)}")
    if "max_items" in exp and len(items) > exp["max_items"]:
        errors.append(f"expected <= {exp['max_items']} items, got {len(items)}")
    if "room_number" in exp and rec.get("room_number") != exp["room_number"]:
        errors.append(f"room_number: expected {exp['room_number']}, got {rec.get('room_number')}")

    for spec in exp.get("must_contain", []):
        if not any(match_item(i, spec) for i in items):
            errors.append(f"must_contain not found: {spec}")
    for spec in exp.get("must_not_contain", []):
        if any(match_item(i, spec) for i in items):
            errors.append(f"must_not_contain found: {spec}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expectations", type=Path, help="expectation JSON file")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    expectations = json.loads(args.expectations.read_text())
    records = load_records(args.data_dir)

    total_failed = 0
    for key, exp in expectations.items():
        recs = find_records(records, key, exp)
        if not recs:
            print(f"❌ {key}: no record found (submit assets/images/{key}.jpeg first)")
            total_failed += 1
            continue
        for rec in recs:
            errors = check_record(rec, exp)
            label = f"{key} ({rec['id'][:8]})"
            if errors:
                total_failed += 1
                print(f"❌ {label}")
                for e in errors:
                    print(f"   - {e}")
            else:
                types = ", ".join(f"{i['type']}({i['class']})" for i in rec["items"])
                print(f"✅ {label}: {len(rec['items'])} items — {types or 'none'}")

    print(f"\n{'PASS' if total_failed == 0 else 'FAIL'}: "
          f"{total_failed} failing record(s) out of {len(expectations)} expectation(s)")
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
