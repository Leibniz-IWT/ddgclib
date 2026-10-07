#!/usr/bin/env bash
# usage: bash docs_temp/workflows/lane_diff.sh [--stat]
#
# Diff of the current (uncommitted) lane against HEAD in ddgclib and in
# hyperct.  Every finished lane is committed, so HEAD is the pre-lane state.
# Unrelated uncommitted work (benchmarks, cases_mean_flow, the other agent's
# capillary-rise files, the user's .gitignore) is excluded.
#
# Paths are derived from this script's location: <ddgclib>/docs_temp/workflows/.
# hyperct is the repo behind the ./hyperct symlink (or ../hyperct next to ddgclib).
DDG=$(cd "$(dirname "$0")/../.." && pwd)
if [ -L "$DDG/hyperct" ]; then
  HYP=$(cd "$(readlink -f "$DDG/hyperct")/.." && pwd)
else
  HYP=$(cd "$DDG/../hyperct" 2>/dev/null && pwd)
fi
EX=(':!cases_mean_flow' ':!benchmarks' ':!.gitignore'
    ':!ddgclib/tests/test_integrated_validation.py'
    ':!cases_dynamic/capillary_rise' ':!cases_dynamic/capillary_rise_energy_grad')
cd "$DDG" || exit 1
echo "##### ddgclib ($DDG, tracked changes vs HEAD)"
if [ "$1" = "--stat" ]; then
  git status --short -- ddgclib cases_dynamic docs_temp METHODS.md debugging_plan.md "${EX[@]}"
else
  git --no-pager diff HEAD -- . "${EX[@]}"
  echo "##### ddgclib (new files)"
  git ls-files --others --exclude-standard -- ddgclib cases_dynamic docs_temp "${EX[@]}" \
    | grep -E '\.(py|md)$' | while read -r f; do
      git --no-pager diff --no-index -- /dev/null "$f"
    done
fi
if [ -z "$HYP" ] || [ ! -d "$HYP/.git" ]; then
  echo "##### hyperct repo not found next to ddgclib; skipped"; exit 0
fi
cd "$HYP" || exit 1
echo "##### hyperct ($HYP, tracked changes vs HEAD)"
if [ "$1" = "--stat" ]; then
  git status --short -- hyperct
else
  git --no-pager diff HEAD -- hyperct
  echo "##### hyperct (new files)"
  git ls-files --others --exclude-standard -- hyperct | grep -E '\.(py|md)$' | while read -r f; do
    git --no-pager diff --no-index -- /dev/null "$f"
  done
fi
exit 0
