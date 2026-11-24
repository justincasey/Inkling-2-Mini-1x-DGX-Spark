#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Mechanical check of docs/PORT-SPEC.md and docs/NUMERICS.md (plan steps 3.1 and 3.2).

The format contract (given to the spec lane, exec plan "Lanes and agents"):

  PORT-SPEC.md  the section headings below, verbatim; section 0 is a table whose last column is a 40-hex revision; every
                `[key]` citation in the body names a section 0 key; section 9 defines `UNKNOWN-<n>:` for n = 1..N
                without gaps or repeats.
  NUMERICS.md   the headings below, verbatim; `## UNKNOWN resolutions` holds exactly one `RESOLVED UNKNOWN-<n>:` or
                `OPEN UNKNOWN-<n>:` line for every id defined in section 9, and none for any other id.

It reads the files and nothing else: no network, no randomness. Prints `check_spec: PASS (...)` and exits 0, or one
`check_spec: FAIL <reason>` line per error and exits 1. `--root DIR` selects the recipe root (default: the parent of
scripts/, or $RECIPE_ROOT).
"""
import argparse
import os
import re
import sys
from pathlib import Path

SPEC_HEADINGS = [
    "## 0 Sources and citation keys",
    "## 1 Global facts",
    "## 2 Notation, dtypes and the TP1 layout",
    "## 3 Target text model, op by op",
    "## 4 Block drafter",
    "## 5 Vision tower",
    "## 6 KV and state per token",
    "## 7 Chat template, reasoning and tool calls",
    "## 8 REXL3 reuse inventory",
    "## 9 UNKNOWNs",
]
NUMERICS_HEADINGS = [
    "## UNKNOWN resolutions",
    "## Other fixed choices",
    "## Rounding points (in forward order)",
    "## Engine emulation",
    "## Drafter rounding points",
]

HEX40 = re.compile(r"^[0-9a-f]{40}$")
BULLET = r"^\s*(?:[-*+]\s+|\d+[.)]\s+)?(?:\*\*|__)?"
END = r"(?::|:\*\*|\*\*:|:__|__:)"
UNKNOWN_DEF = re.compile(BULLET + r"UNKNOWN-(\d+)" + END)
RESOLUTION = re.compile(BULLET + r"(RESOLVED|OPEN)\s+UNKNOWN-(\d+)" + END)
CITATION = re.compile(r"\[([A-Za-z][A-Za-z0-9_.\-]{2,})\]")


def sections(text):
    """Map each '## ' heading line (stripped) to its body lines (up to the next '## ' heading)."""
    out, cur = {}, None
    for line in text.split("\n"):
        if line.startswith("## "):
            cur = line.strip()
            out.setdefault(cur, [])
        elif cur is not None:
            out[cur].append(line)
    return out


def strip_code(text):
    text = re.sub(r"^```.*?^```[ \t]*$", "", text, flags=re.S | re.M)
    text = re.sub(r"`[^`\n]*`", "", text)
    return re.sub(r"\[[^\]\n]*\]\([^)\n]*\)", "", text)


def table_rows(lines):
    rows = []
    for line in lines:
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
            continue
        rows.append(cells)
    return rows


def ids_in(lines, pattern, group):
    return [int(m.group(group)) for line in lines for m in [pattern.match(line)] if m]


def check(root):
    errors = []
    spec_path, num_path = root / "docs" / "PORT-SPEC.md", root / "docs" / "NUMERICS.md"
    for p in (spec_path, num_path):
        if not p.is_file():
            errors.append(f"{p.relative_to(root)} does not exist")
    if errors:
        return errors, ""
    spec, num = spec_path.read_text(encoding="utf-8"), num_path.read_text(encoding="utf-8")
    spec_sec, num_sec = sections(spec), sections(num)
    for h in SPEC_HEADINGS:
        if h not in spec_sec:
            errors.append(f"PORT-SPEC.md lacks the heading '{h}'")
    for h in NUMERICS_HEADINGS:
        if h not in num_sec:
            errors.append(f"NUMERICS.md lacks the heading '{h}'")

    keys = {}
    header_seen = False
    for cells in table_rows(spec_sec.get(SPEC_HEADINGS[0], [])):
        if not header_seen:  # the table's header row
            header_seen = True
            continue
        key = re.sub(r"[`*\[\]]", "", cells[0]).strip()
        rev = re.sub(r"[`*]", "", cells[-1]).strip()
        if not key:
            continue
        if key in keys:
            errors.append(f"section 0: duplicate citation key '{key}'")
        if not HEX40.match(rev):
            errors.append(f"section 0: key '{key}' has no 40-hex revision in its last column ('{rev[:40]}')")
        keys[key] = rev
    if SPEC_HEADINGS[0] in spec_sec and not keys:
        errors.append("section 0 has no table rows")

    body = strip_code("\n".join("\n".join(v) for k, v in spec_sec.items() if k != SPEC_HEADINGS[0]))
    for cite in sorted(set(CITATION.findall(body))):
        if cite not in keys:
            errors.append(f"PORT-SPEC.md cites [{cite}], which section 0 does not define")

    defined = ids_in(spec_sec.get(SPEC_HEADINGS[9], []), UNKNOWN_DEF, 1)
    if len(defined) != len(set(defined)):
        errors.append("section 9 defines an UNKNOWN id twice")
    if sorted(set(defined)) != list(range(1, len(set(defined)) + 1)):
        errors.append(f"section 9 ids are not 1..N without gaps: {sorted(set(defined))}")

    res_lines = num_sec.get(NUMERICS_HEADINGS[0], [])
    kinds = {}
    for line in res_lines:
        m = RESOLUTION.match(line)
        if m:
            kinds.setdefault(int(m.group(2)), []).append(m.group(1))
    for i in sorted(set(defined)):
        n = len(kinds.get(i, []))
        if n != 1:
            errors.append(f"NUMERICS.md has {n} RESOLVED/OPEN lines for UNKNOWN-{i} (need exactly 1)")
    for i in sorted(set(kinds) - set(defined)):
        errors.append(f"NUMERICS.md resolves UNKNOWN-{i}, which PORT-SPEC.md section 9 does not define")

    resolved = sum(1 for v in kinds.values() if v == ["RESOLVED"])
    open_ = sum(1 for v in kinds.values() if v == ["OPEN"])
    return errors, f"keys={len(keys)}, unknowns={len(set(defined))}, resolved={resolved}, open={open_}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.environ.get("RECIPE_ROOT") or str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()
    errors, summary = check(Path(args.root))
    if errors:
        for e in errors:
            print(f"check_spec: FAIL {e}")
        return 1
    print(f"check_spec: PASS ({summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
