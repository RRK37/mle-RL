"""
Setup module for AIDE (AI Development Environment) agent.

This module provides functions to set up and configure the AIDE agent
for Kaggle competitions. It handles loading configurations, preparing
workspaces, and initializing the agent with appropriate parameters.

The module is designed to work with the main.py script and integrates
with the MLE-Dojo framework for running AI agents on Kaggle competitions.
"""

import os
import logging
from typing import Dict, Any, Tuple

from mledojo.agent.aide.agent import Agent
from mledojo.agent.aide.journal import Journal
from mledojo.agent.aide.utils.config import (
    _load_cfg,
    prep_cfg,
    load_task_desc,
    prep_agent_workspace,
)
from mledojo.utils import get_metric

from rich.status import Status

logger = logging.getLogger("aide_setup")

def setup_aide_agent(
    config: Dict[str, Any], 
) -> Tuple[Agent, Journal, Any]:
    """
    Set up an AIDE Agent based on configuration.
    
    This function initializes and configures an AIDE agent for a specific
    Kaggle competition. It loads the configuration, prepares the workspace,
    and initializes the agent with the appropriate parameters.
    
    Args:
        config: Configuration dictionary containing agent and competition settings
        competition_name: Name of the Kaggle competition
        output_dir: Directory where outputs will be stored
        
    Returns:
        Tuple containing:
            - Configured AIDE Agent instance
            - Journal for tracking agent progress
            - Configuration object for the agent
    """
    # Load and prepare AIDE config
    _cfg = _load_cfg(use_cli_args=False)
    data_dir = config['competition']['data_dir']
    desc_file = os.path.join(data_dir, "public", "description.txt")
    
    # Configure paths and settings
    _cfg.data_dir = os.path.join(data_dir, "public")
    _cfg.name = config['competition']['name']
    _cfg.desc_file = desc_file
    _cfg.log_dir = os.path.join(config['output_dir'], "logs")
    _cfg.workspace_dir = config['output_dir']
    
    # Apply agent configuration from the config file
    if 'agent' in config and config['agent']:
        agent_cfg = config['agent']
        
        # Apply top-level agent settings
        if 'steps' in agent_cfg:
            _cfg.agent.steps = agent_cfg['steps']
        if 'k_fold_validation' in agent_cfg:
            _cfg.agent.k_fold_validation = agent_cfg['k_fold_validation']
        if 'expose_prediction' in agent_cfg:
            _cfg.agent.expose_prediction = agent_cfg['expose_prediction']
        if 'data_preview' in agent_cfg:
            _cfg.agent.data_preview = agent_cfg['data_preview']
        
        # Apply code generation settings
        if 'code' in agent_cfg:
            code_cfg = agent_cfg['code']
            for key, value in code_cfg.items():
                if hasattr(_cfg.agent.code, key):
                    setattr(_cfg.agent.code, key, value)
        
        # Apply search settings
        if 'search' in agent_cfg:
            search_cfg = agent_cfg['search']
            for key, value in search_cfg.items():
                if hasattr(_cfg.agent.search, key):
                    setattr(_cfg.agent.search, key, value)
    
    # Prepare the configuration
    cfg = prep_cfg(_cfg)
    task_desc = load_task_desc(cfg)

    # Prepare the agent workspace
    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)
    
    # Initialize the journal and get metric information
    journal = Journal()
    metric_class = get_metric(config['competition']['name'])
    higher_is_better = metric_class().higher_is_better
    
    # Create the agent
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
        higher_is_better=higher_is_better,
        data_dir=os.path.join(cfg.workspace_dir, "input"),
        output_dir=cfg.workspace_dir
    )
    
    logger.info(f"AIDE agent set up for competition: {config['competition']['name']}")
    return agent, journal, cfg
