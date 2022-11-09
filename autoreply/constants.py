from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
assert ROOT.exists()
DATA = ROOT.joinpath('data')
