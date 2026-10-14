"""Generate examples/old.step and examples/new.step (demo scenario with known ground truth)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from step_diff.samples import write_demo  # noqa: E402

if __name__ == "__main__":
    old, new, expected = write_demo(HERE)
    (HERE / "expected.json").write_text(json.dumps(expected, indent=2))
    print(f"written: {old}\n         {new}\n         {HERE / 'expected.json'}")
