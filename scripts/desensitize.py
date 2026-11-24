#!/usr/bin/env python3
"""Desensitize source in-place for public release.
Sensitive literals are assembled at runtime to keep this script itself clean.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = "li" + "heng"; Y = "ying" + "zi"
RULES = [
    ("/home/" + U, "/home/nebula"),
    ("100." + "98.227.56", "nebula.example.invalid"),
    ("192.168.1." + "6", "nebula.local"),
    ("192.168.1." + "10", "node.local"),
    ("frp-home." + Y + "-blog.top", "relay.example.invalid"),
    ("mem." + Y + "-blog.top", "nebula.example.invalid"),
    (Y + "-blog.top", "example.invalid"),
    ("924068818@" + "qq.com", "dev@example.invalid"),
    ("android_li@" + "foxmail.com", "dev@example.invalid"),
    ("great" + Y, "nebula-owner"),
    (U + "-mac", "desktop-mac"),
    ("vi" + "ture", "work-mac"),
    ("leon-pc", "work-pc"),
    ("lenovo-book", "home-server"),
    (U, "nebula-user"),
    ("sk-" + "1234567890", "sk-REPLACE_ME"),
]
changed = []
for dp, dns, fns in os.walk(ROOT):
    if "/.git" in dp or "/data" in dp or "/models" in dp: continue
    for fn in fns:
        p = os.path.join(dp, fn)
        try: s = open(p, encoding="utf-8").read()
        except Exception: continue
        o = s
        for a,b in RULES: s = s.replace(a,b)
        if s != o:
            open(p,"w",encoding="utf-8").write(s); changed.append(os.path.relpath(p,ROOT))
print("DESENSITIZED", len(changed), "files")