from gymnasium.envs.registration import register

register(
     id="SamSegEnv-v0",
     entry_point="custom_gym_implns.envs:SamSegEnv",
)
# Registers a Gymnasium environment for interactive SAM segmentation.
# The environment exposes a discrete action space over point prompts and returns observations
# containing SAM features and current mask probability, aligning with the AlignSAM training setup.