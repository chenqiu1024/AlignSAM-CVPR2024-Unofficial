from models.implicit_agent import ImplicitAgent
from models.explicit_agent import ExplicitAgent


def make_agent(agent_cfg, envs):
    # Factory to select between:
    # - ImplicitAgent: uses only SAM features (no CLIP). Baseline aligning SAM via RL.
    # - ExplicitAgent: fuses SAM with CLIP-surgery features guided by text prompts, per AlignSAM.
    if agent_cfg['type'] == "implicit":
        return ImplicitAgent(envs, agent_cfg)
    elif agent_cfg['type'] == "explicit":
        return ExplicitAgent(envs, agent_cfg)
    else:
        raise ValueError(f"Unknown agent type {agent_cfg['type']}")