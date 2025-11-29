import json

class GEPAAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.prompt_template = """
You are a data scientist competing in a Kaggle competition.
Your goal is to achieve the highest score possible.
Here is the information about the competition: {competition_info}
Here is the history of your previous attempts: {history}
Here is the feedback from your last attempt: {feedback}
Your current score is: {score}

Based on this information, what is your next action?
Your action should be a JSON object with two keys: 'action' and 'code'.
The 'action' should be one of: 'request_info', 'validate_code', 'execute_code'.
The 'code' should be the Python code you want to execute.
"""

    def _construct_prompt(self, observation):
        # For now, we'll use placeholder values for the prompt.
        # In a real implementation, you would extract this information
        # from the observation dictionary.
        prompt = self.prompt_template.format(
            competition_info="Titanic: Predict survival on the Titanic.",
            history=observation.get("history_summary", "No history yet."),
            feedback=observation.get("feedback", "No feedback yet."),
            score=observation.get("current_raw_score", 0)
        )
        return prompt

    def select_action(self, observation):
        prompt = self._construct_prompt(observation)
        
        # In a real implementation, you would call the LLM here.
        # llm_response = self.llm_client.generate(prompt)
        
        # Hardcoded response for testing
        llm_response = '{"action": "execute_code", "code": "print(\'hello world\')"}'
        
        try:
            action_json = json.loads(llm_response)
        except json.JSONDecodeError:
            # Handle cases where the LLM response is not valid JSON
            print("Error: LLM response is not valid JSON.")
            return {"action": "get_history", "code": ""}

        return action_json

    def update(self, obs, action, reward, next_obs, done):
        # Not needed for this simple agent
        pass
