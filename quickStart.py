import sys
from pathlib import Path
# Add submodules/mle-dojo to the Python path
mle_dojo_path = Path(__file__).parent / "submodules" / "mle-dojo"
sys.path.insert(0, str(mle_dojo_path))
from mledojo.gym.competition import CompetitionRegistry, CompInfo
from mledojo.competitions import get_metric
from mledojo.gym.env import KaggleEnvironment


