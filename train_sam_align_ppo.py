# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
#
# AlignSAM context:
# - This training script implements on-policy reinforcement learning (PPO) to learn a policy that
#   interactively prompts SAM (Segment Anything Model) for target-aware segmentation.
# - The environment provides SAM image embeddings and current mask probability; the agent outputs
#   a discrete action corresponding to a click location and label (positive/negative), aligning
#   with the promptable segmentation interface described in Segment Anything (Kirillov et al., 2023).
# - The overall idea follows AlignSAM: Aligning Segment Anything Model to Open Context via
#   Reinforcement Learning (CVPR 2024 workshop) where PPO is used to train an agent to adapt SAM
#   using interaction signals and reward shaped by segmentation quality.
#   See: docs/AlignSAM- Aligning Segment Anything Model to Open Context via Reinforcement Learning.pdf
#   See: docs/Segment Anything.pdf
# - PPO details follow the standard formulation from Schulman et al., 2017.
#   See: clipped objective, advantage estimation, and value loss below.
import os
import random
import time
from dataclasses import dataclass

# Training entrypoint for AlignSAM with PPO (on-policy RL).
#
# High-level overview and references (see `docs/`):
# - AlignSAM: Aligning Segment Anything Model to Open Context via Reinforcement Learning.
#   This script implements the RL loop where the agent learns to place point prompts
#   to guide SAM toward a target category mask.
# - Segment Anything: Provides the promptable segmentation interface (points) used by the env.
# - A Closer Look at CLIP Explainability: Supplies token-level similarity maps used by the
#   explicit agent to ground clicks in text semantics (via CLIP-Surgery utilities).
# - PPO: Uses clipped surrogate objective, GAE(λ), and value loss per standard practice.

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import mlflow
import yaml
from functools import partial

from models import make_agent

import custom_gym_implns


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    mlflow_project_name: str = "sam_align"
    """the MLFlow's project name"""
    mlflow_user: str = None
    """the username (team) of MLFlow's project"""
    log_dir: str = "logs"
    """the logging directory for the experiment"""
    resume_from: str = None
    """the path to the checkpoint for resuming the training"""
    checkpoint_iter_freq: int = 50
    """the frequency of making checkpoint (if applicable)"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `{log_dir}/{run_name}/videos` folder)"""
    capture_ep_freq: int = 100
    """the frequency of capturing videos of the agent performances"""

    # Algorithm specific arguments
    env_id: str = "SamSegEnv-v0"
    """the id of the environment"""
    env_cfg_path: str = "configs/envs/repvit_sam_cbseg.yaml"
    """the environment configuration path"""
    agent_cfg_path: str = "configs/agents/explicit_agent.yaml"
    """the type of the agent"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 16
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
# Notes on PPO hyperparameters:
# - clip_coef: surrogate objective clipping parameter per PPO (Schulman et al., 2017)
# - gae_lambda: Generalized Advantage Estimation lambda (Schulman et al., 2016)
# - ent_coef: encourages exploration via policy entropy
# - vf_coef: weight for value loss in the combined objective
# - target_kl: early stopping when empirical KL exceeds threshold

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, capture_ep_freq, log_dir, env_cfg):
    def capture_ep(capture_ep_freq, episode_idx):
        return episode_idx % capture_ep_freq == 0
    
    def thunk():
        # The environment is a SAM-based interactive segmentation task. The agent's action is a
        # click (location + positive/negative label). This encapsulates the prompt design from
        # Segment Anything (Kirillov et al., 2023), see docs/Segment Anything.pdf.
        if capture_video and idx == 0:
            env = gym.make(env_id, **env_cfg)
            video_folder = os.path.join(log_dir, "videos")
            env = gym.wrappers.RecordVideo(env, video_folder, 
                                           episode_trigger=partial(capture_ep, capture_ep_freq))
        else:
            env = gym.make(env_id, **env_cfg)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk



def load_obs_to_tensor(obs, device):
    obs_tensor = dict()
    for key in obs.keys():
        # Observation dictionary contains:
        # - "image": raw RGB (H, W, C)
        # - "sam_image_embeddings": SAM image encoder features (C, H', W')
        # - "sam_pred_mask_prob": current mask probability map (Hmask, Wmask)
        obs_tensor[key] = torch.Tensor(obs[key]).to(device)
    return obs_tensor
## Update one steop of observations with next_obs:
def update_obs_at_step(obs, next_obs, step):
    for key in obs.keys():
        obs[key][step] = next_obs[key]

def flatten_obs(obs):
    new_obs = dict()
    for key, val in obs.items(): ## val's shape is (#steps, #envs, sizeof(env))
        val_shape = list(val.size())
        num_steps, num_envs = val_shape[:2] 
        new_obs[key] = val.reshape(num_steps * num_envs, *val_shape[2:])
    return new_obs
## Pick out all observations of a single step:
def get_obs_at_inds(obs, mb_inds):
    new_obs = dict()
    for key, val in obs.items():
        new_obs[key] = val[mb_inds]
    return new_obs


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not os.path.exists(args.env_cfg_path):
        raise ValueError(f"env_cfg_path {args.env_cfg_path} does not exist")
    
    env_cfg = yaml.safe_load(open(args.env_cfg_path, "r"))

    if not os.path.exists(args.agent_cfg_path):
        raise ValueError(f"agent_cfg_path {args.agent_cfg_path} does not exist")
    
    agent_cfg = yaml.safe_load(open(args.agent_cfg_path, "r"))

    log_dir = os.path.join(args.log_dir.rstrip("/"), run_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if (args.resume_from is not None) and (not os.path.exists(args.resume_from)):
        raise ValueError(f"resume_from {args.resume_from} does not exist")
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    mlflow_run_tags = {
        "user": args.mlflow_user,
        "project": args.mlflow_project_name,
    }

    with mlflow.start_run(run_name=run_name, tags=mlflow_run_tags, description="parent") as parent_run:
        writer = SummaryWriter(log_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        # env setup
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, i, args.capture_video, args.capture_ep_freq, log_dir, env_cfg) \
             for i in range(args.num_envs)],
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        # The discrete action space indexes a pre-defined grid of clicks with positive/negative label.
        # See `SamSegEnv` for the mapping logic.

        agent = make_agent(agent_cfg, envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        if args.resume_from is not None:
            checkpoint = torch.load(args.resume_from)
            agent.load_state_dict(checkpoint["model"])
            optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
            optimizer.load_state_dict(checkpoint["optimizer"])

        # ALGO Logic: Storage setup
        # PPO collects trajectories over num_steps and num_envs, then optimizes policy and value nets
        # using minibatch SGD on the clipped objective.
        obs = dict()
        # obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        for key, space in envs.single_observation_space.spaces.items():
            obs[key] = torch.zeros((args.num_steps, args.num_envs) + space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        raw_next_obs, _ = envs.reset(seed=args.seed)
        next_obs = load_obs_to_tensor(raw_next_obs, device)
        # next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            # Linear LR annealing per CleanRL PPO recipe.
            if args.anneal_lr:
                frac = 1.0 - ((iteration - 1.0) / args.num_iterations)
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                update_obs_at_step(obs, next_obs, step)
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    # Actor-critic forward: outputs action distribution and state value.
                    # The agent merges SAM features with CLIP-surgery features (explicit agent) or
                    # uses only SAM features (implicit agent). See models/explicit_agent.py and
                    # models/implicit_agent.py for details and citations.
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_raw_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations) ## Combines terminal and time-limit endings into a single done mask.
                rewards[step] = torch.tensor(reward).to(device).view(-1) ## view(-1) flatten to a 1D tensor with inferred size, matching the per-step rewards shape.
                next_obs = load_obs_to_tensor(next_raw_obs, device)
                next_done = torch.Tensor(next_done).to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                # Generalized Advantage Estimation (GAE; Schulman et al., 2016):
                # advantages[t] = δ_t + γλ δ_{t+1} + (γλ)^2 δ_{t+2} + ...
                # where δ_t = r_t + γ V(s_{t+1}) - V(s_t)
                # Using γ=0.99 and λ=0.95 per common PPO settings stabilizes training.
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    # GAE(λ) temporal-difference residual (Schulman et al., 2016)
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_obs = flatten_obs(obs)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    mb_obs = get_obs_at_inds(b_obs, mb_inds)

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, 
                                                                                  b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss (PPO clipped surrogate objective; Schulman et al., 2017)
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss (optionally clipped)
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        # Clipped value objective mirrors the policy clip to reduce value overfitting.
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            mlflow.log_metric("losses/value_loss", v_loss.item(), global_step)
            mlflow.log_metric("losses/policy_loss", pg_loss.item(), global_step)
            mlflow.log_metric("losses/entropy", entropy_loss.item(), global_step)


            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            print("Iteration:", iteration)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            # In AlignSAM terms, higher episodic returns imply improved alignment of interactive
            # prompts to produce target-specific masks. See docs/AlignSAM- Aligning Segment Anything Model to Open Context via Reinforcement Learning.pdf
            # The RL loop aligns the agent's click strategy with segmentation quality, while SAM
            # executes the promptable segmentation per docs/Segment Anything.pdf.

            if iteration % args.checkpoint_iter_freq == 0:
                checkpoint = {
                    "model": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pth"))
                latest_symbolic = os.path.join(checkpoint_dir, "latest.pth")
                if os.path.islink(latest_symbolic):
                    os.unlink(latest_symbolic)
                os.symlink(f"checkpoint_{iteration}.pth", latest_symbolic)

        envs.close()
        writer.close()

        print("Uploading TensorBoard events as a run artifact...")
        mlflow.log_artifacts(log_dir, artifact_path="events")
        print(
            "\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s"
            % os.path.join(mlflow.get_artifact_uri(), "events")
        )