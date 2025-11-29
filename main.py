import os
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, './submodules/mle-dojo')

from gepa_agent import GEPAAgent
from mledojo.gym.competition import CompetitionRegistry
from mledojo.gym.env import KaggleEnvironment

# Mock LLM client for testing
class MockLLMClient:
    def generate(self, prompt):
        return '{"action": "execute_code", "code": "print(\'hello world\')"}'

def main():
    # Create a temporary directory for the output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        # 1. Instantiate CompetitionRegistry
        # It will automatically discover the competitions in the submodules/mle-dojo/mledojo/competitions directory
        registry = CompetitionRegistry()

        # 2. Instantiate KaggleEnvironment
        env = KaggleEnvironment.make(
            competition_name="titanic",
            output_dir=output_dir,
            competition_registry=registry
        )

        # 3. Instantiate GEPAAgent
        llm_client = MockLLMClient()
        agent = GEPAAgent(llm_client)

        # 4. Run a simple loop
        obs, _ = env.reset()
        action_json = agent.select_action(obs)
        
        # The step function expects the action name and kwargs
        action_name = action_json["action"]
        action_kwargs = {"code": action_json["code"]}

        obs, reward, terminated, truncated, info = env.step(action_name, **action_kwargs)

        # 5. Print the results
        print("Observation:", obs)
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)
        print("Info:", info)

if __name__ == "__main__":
    main()
