#!/usr/bin/env bash
# Check the UI's Datastar attributes against the runtime vendored beside it.
set -e
node "$(cd "$(dirname "$0")" && pwd)/datastar.test.js"
