#!/bin/zsh
set -e

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

if command -v /usr/local/bin/python3 >/dev/null 2>&1; then
  PYTHON=/usr/local/bin/python3
else
  PYTHON=python3
fi

LOG="$SCRIPT_DIR/recompute_all_cases_terminal.log"
set +e
"$PYTHON" "$SCRIPT_DIR/recompute_all_cases.py" 2>&1 | tee "$LOG"
status=${pipestatus[1]}
set -e

echo
if [ "$status" -eq 0 ]; then
  echo "Figure 3-8 recomputation finished. Log: $LOG"
  echo "You may close this window."
else
  echo "Figure 3-8 recomputation failed. Log: $LOG"
  exit "$status"
fi
