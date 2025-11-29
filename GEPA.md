# GEPA: Genetic-Pareto Prompt Optimizer

GEPA (Genetic-Pareto Prompt Optimizer) is a powerful and efficient technique for optimizing prompts for Large Language Models (LLMs) and other AI systems. It was proposed in the paper "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" (Agrawal et al., 2025).

GEPA uses an evolutionary algorithm to iteratively improve prompts based on natural language reflection and multi-objective selection. This approach has been shown to outperform traditional reinforcement learning (RL) methods in terms of both performance and sample efficiency.

## Key Concepts

The core of GEPA lies in its "reflective prompt evolution" mechanism, which leverages the following key concepts:

*   **Natural Language Reflection:** Instead of relying on sparse, scalar rewards like traditional RL, GEPA uses natural language feedback to guide the optimization process. An LLM is used to analyze the execution traces, reasoning paths, and even errors of the system to diagnose what went wrong and why.
*   **Evolutionary Algorithm:** GEPA uses a genetic algorithm to evolve a population of prompts over multiple generations. This involves selection, mutation, and crossover of prompts to generate new and improved candidates.
*   **Pareto-based Selection:** GEPA uses a multi-objective selection strategy based on Pareto fronts. This allows it to maintain a diverse pool of high-performing prompts and avoid getting stuck in local optima.

## How it Works

The GEPA workflow can be summarized as follows:

1.  **Initialization:** The process starts with a "seed" prompt, which can be a simple, hand-crafted prompt.
2.  **Evaluation:** The current generation of prompts is evaluated against a dataset. The outputs of the system are scored by an evaluator.
3.  **Reflection:** A powerful "reflection LLM" analyzes the evaluation results, particularly the failures, and generates detailed textual feedback. This feedback is used to create a "reflective dataset."
4.  **Evolution (Mutation):** The reflective dataset guides the evolution process. The reflection model generates a new population of candidate prompts (mutations) that are designed to avoid the failures of the previous generation.
5.  **Selection:** The new prompts are evaluated, and the best-performing ones are selected to form the next generation. This process is repeated until a predefined budget is exhausted.

## Advantages over Reinforcement Learning

GEPA offers several advantages over traditional RL-based methods for prompt optimization:

*   **Sample Efficiency:** GEPA requires significantly fewer rollouts (up to 35x fewer) to achieve substantial improvements, making it more cost-effective and faster.
*   **Interpretability:** The use of natural language reflection makes the optimization process more transparent and easier to debug.
*   **Framework-Agnostic:** GEPA is designed to work across various AI frameworks.
*   **Domain Adaptability:** It excels at incorporating domain-specific knowledge through textual feedback.

## Application to `mle-dojo`

The `mle-dojo` environment is an ideal use case for GEPA. The `KaggleEnvironment` is designed to be controlled by an agent that generates code, which is a natural fit for an LLM-based agent.

Here's how GEPA could be applied to the `mle-dojo` environment:

1.  **LLM-based Agent:** An LLM-based agent would be used to interact with the `KaggleEnvironment`. The agent would take the observation from the environment (feedback, scores, history) and use it to construct a prompt for the LLM.
2.  **Prompt Structure:** The prompt would be designed to instruct the LLM to act as a data scientist competing in a Kaggle competition. It would include placeholders for the competition information, history, feedback, and score.
3.  **GEPA for Prompt Optimization:** GEPA would be used to evolve the prompt. The fitness of each prompt would be determined by the final score achieved by the agent in the `KaggleEnvironment`.

By using GEPA to optimize the prompt, the LLM-based agent could learn to generate better and better code, leading to higher scores in the competition.