#!/usr/bin/env bash
# Run the whole suite. Uses stubbed models, so it needs no weights and no GPU.
set -u
cd "$(dirname "$0")"
PY=./.venv/bin/python
fail=0
for t in tests/test_integration.py tests/test_cli.py tests/test_concurrency.py tests/test_repl.py; do
  echo "=============================================="
  echo "  $t"
  echo "=============================================="
  $PY "$t" || fail=1
done
if command -v node >/dev/null 2>&1; then
  echo "=============================================="
  echo "  tests/web/md.test.js"
  echo "=============================================="
  ./tests/web/run.sh || fail=1
else
  echo "(skipping web UI tests: node not found)"
fi

echo
if [ $fail -eq 0 ]; then echo "ALL SUITES PASSED"; else echo "SOME SUITES FAILED"; fi
exit $fail
