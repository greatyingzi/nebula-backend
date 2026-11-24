#!/usr/bin/env python3
"""Desensitization gate: fail if any sensitive pattern is present in tracked files.
Patterns assembled at runtime so this script stays clean.
"""
import os, sys, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = "li" + "heng"; Y = "ying" + "zi"
PATTERNS = [
    U, Y, "100" + r"\.98\.", "192" + r"\.168\.", "frp-home",
    "924068818|foxmail" + "|qq" + r"\.com", "sk-" + "[a-zA-Z0-9]{8,}",
]
FOUND = []
for dp, dns, fns in os.walk(ROOT):
    if "/.git" in dp or "/data" in dp or "/models" in dp or "/scripts" in dp: continue
    for fn in fns:
        p = os.path.join(dp, fn)
        try: lines = open(p, encoding="utf-8").read().splitlines()
        except Exception: continue
        for i, line in enumerate(lines, 1):
            for pat in PATTERNS:
                if re.search(pat, line, re.I):
                    if "example.invalid" in line: continue
                    if "github.com/" + "great" + Y + "/" in line: continue  # public repo link
                    FOUND.append(f"{os.path.relpath(p,ROOT)}:{i}: {line.strip()[:120]}")
                    break
if FOUND:
    print("DESENSITIZATION GATE FAILED:")
    for f in FOUND[:50]: print("  ", f)
    sys.exit(1)
print("DESENSITIZATION GATE PASS")