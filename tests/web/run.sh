#!/usr/bin/env bash
# Extract the pure helpers from the single-file UI and test them under node.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/../.."
python3 - "$root" "$here" <<'PY'
import re, sys, pathlib
root, here = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
html = (root / "src/hearth/web/index.html").read_text()
js = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
# Everything up to the rendering section is DOM-free and testable as-is.
js = js[: js.index("// ---------------- rendering")]
(here / "md.js").write_text(js + "\nmodule.exports = { md, esc };\n")
PY
node "$here/md.test.js"
