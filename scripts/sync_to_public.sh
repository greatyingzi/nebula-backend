#!/bin/bash
# sync_to_public.sh — 从内部仓同步 Rust 核心到公开仓并脱敏
set -euo pipefail
SRC="${1:-$HOME/linghub}"
PUB="$(cd "$(dirname "$0")/.." && pwd)"
echo "[1/4] copy rust core"
rm -rf "$PUB/rust"; mkdir -p "$PUB/rust"
(cd "$SRC/rust" && tar czf - --exclude=target --exclude='*.bak*' --exclude=.git \
  Cargo.toml Cargo.lock nebula-sync-rs nebula-mcp-rs) | tar xzf - -C "$PUB/rust"
echo "[2/5] trim workspace members (only public crates)"
python3 - "$PUB/rust/Cargo.toml" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
s=s.replace('members = ["nebula-sync-rs", "archive-local-rs", "linghub-daemon-rs", "nebula-mcp-rs"]','members = ["nebula-sync-rs", "nebula-mcp-rs"]')
open(p,'w').write(s)
PY
echo "[3/5] desensitize"
python3 "$PUB/scripts/desensitize.py"
echo "[4/5] gate"
python3 "$PUB/scripts/desens_gate.py"
echo "[5/5] done; review then commit"