import json
import os
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def pool(tmp_path_factory):
    """A pool of random legal teams, from the engine's own sampler (cargo build --release first)."""
    sample = subprocess.run([os.path.join(REPO, "target", "release", "teamcheck"), "--sample", "24", "--seed", "5"],
                            capture_output=True, text=True, check=True).stdout
    teams = [{"player": f"player {k}", "team": json.loads(line)} for k, line in enumerate(sample.splitlines()) if line.strip()]
    path = tmp_path_factory.mktemp("pool") / "pool.json"
    path.write_text(json.dumps({"source": "", "event": "made up", "format": "gen9championsvgc2026regmc", "teams": teams}))
    return str(path)
