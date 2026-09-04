"""Run the real server with the two model engines stubbed out.

Used by test_cli.py so the CLI can be exercised over real HTTP without
loading 60GB of weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_integration_stubs import fake_image_stream, fake_text_stream  # noqa: E402

from hearth.engine.image import ImageEngine  # noqa: E402
from hearth.engine.text import TextEngine  # noqa: E402

TextEngine.stream = fake_text_stream
ImageEngine.stream = fake_image_stream

if __name__ == "__main__":
    import uvicorn

    from hearth import config as config_mod
    from hearth.server import create_app

    cfg = config_mod.load()
    uvicorn.run(create_app(cfg), host="127.0.0.1", port=cfg.server.port, log_level="error")
