"""Create the canonical rule lookup table for the fish case study.

Every downstream script reads its labels from data/fish/rule_lookup.csv so
that the integer <-> rule mapping is defined exactly once. The columns are:

    rule_id           0..10, the primary 11-class target
    rule_name         human-readable name used in figures and tables
    strategy_family   none | nearest | random | mostinfluential | all
    family_id         0..4, the secondary target for RQ2
    neighbor_count    k in 0..4, the tertiary target for RQ2
    source_file       the .dat file the runs come from

Run from the repository root:
    python scripts/fish/02_create_rule_lookup.py
"""

import csv
import json
import os

INSPECTION_PATH = os.path.join("results", "fish", "inspection.json")
OUT_PATH = os.path.join("data", "fish", "rule_lookup.csv")

# Order matters: family_id is the index into this list.
FAMILY_ORDER = ["none", "nearest", "random", "mostinfluential", "all"]

# Display names for figures. Keyed by (family, k).
DISPLAY_NAME = {
    ("none", 0): "no-interaction",
    ("nearest", 1): "nearest-1",
    ("nearest", 2): "nearest-2",
    ("nearest", 3): "nearest-3",
    ("random", 1): "random-1",
    ("random", 2): "random-2",
    ("random", 3): "random-3",
    ("mostinfluential", 1): "influential-1",
    ("mostinfluential", 2): "influential-2",
    ("mostinfluential", 3): "influential-3",
    ("all", 4): "all-4",
}


def main():
    if not os.path.exists(INSPECTION_PATH):
        raise SystemExit(
            "Missing %s. Run 01_inspect_agent_fish_data.py first." % INSPECTION_PATH)

    with open(INSPECTION_PATH) as fh:
        inspection = json.load(fh)

    # Invert the per-file report: rule_id -> the file that produced it. This
    # keeps the lookup table tied to the data actually on disk rather than to
    # a hard-coded list that could drift away from it.
    by_rule = {}
    for fname, rep in inspection["per_file"].items():
        rid = rep["rule_id"]
        if rid in by_rule:
            raise SystemExit("two files claim rule_id %d: %s and %s"
                             % (rid, by_rule[rid]["source_file"], fname))
        by_rule[rid] = {
            "rule_id": rid,
            "strategy_family": rep["strategy_family"],
            "neighbor_count": rep["neighbor_count"],
            "source_file": fname,
            "n_runs": rep["n_experiments"],
        }

    n_rules = inspection["n_rules"]
    missing = sorted(set(range(n_rules)) - set(by_rule))
    if missing:
        raise SystemExit("no file found for rule ids %s" % missing)

    rows = []
    for rid in range(n_rules):
        r = by_rule[rid]
        fam, k = r["strategy_family"], r["neighbor_count"]
        rows.append({
            "rule_id": rid,
            "rule_name": DISPLAY_NAME[(fam, k)],
            "strategy_family": fam,
            "family_id": FAMILY_ORDER.index(fam),
            "neighbor_count": k,
            "source_file": r["source_file"],
            "n_runs": r["n_runs"],
        })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fields = ["rule_id", "rule_name", "strategy_family", "family_id",
              "neighbor_count", "source_file", "n_runs"]
    with open(OUT_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("%-3s %-16s %-16s %-3s %-3s %-6s %s"
          % ("id", "rule_name", "strategy_family", "fid", "k", "runs", "source_file"))
    print("-" * 92)
    for r in rows:
        print("%-3d %-16s %-16s %-3d %-3d %-6d %s"
              % (r["rule_id"], r["rule_name"], r["strategy_family"], r["family_id"],
                 r["neighbor_count"], r["n_runs"], r["source_file"]))

    print("\nPrediction targets")
    print("  exact rule ....... %d classes, chance %.4f"
          % (n_rules, 1.0 / n_rules))
    fams = sorted({r["strategy_family"] for r in rows})
    print("  strategy family .. %d classes (%s)" % (len(fams), ", ".join(fams)))
    ks = sorted({r["neighbor_count"] for r in rows})
    print("  neighbor count ... %d classes (k = %s)"
          % (len(ks), ", ".join(str(k) for k in ks)))

    print("\nNote: family and k are NOT independent. k=0 occurs only with")
    print("'none' and k=4 only with 'all', so those two families are")
    print("determined by k alone. The genuinely ambiguous decisions live")
    print("inside nearest/random/mostinfluential at k in {1,2,3}: 9 of the")
    print("11 rules, and the region where we expect conformal sets to widen.")

    print("\nWrote %s" % OUT_PATH)


if __name__ == "__main__":
    main()
