import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import yaml
import cv2

from functools import partial

# Non-interactive backend for headless environments (e.g., Colab/servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    accelerator: bool = True
    """if toggled, accelerator (gpu, mps etc) will be enabled by default"""

    log_dir: str = "logs"
    """the logging directory for the experiment"""
    resume_from: str = None
    """the path to the checkpoint for resuming the training"""
    checkpoint_iter_freq: int = 1000
    """the frequency of making checkpoint (if applicable)"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `{log_dir}/{run_name}/videos` folder)"""
    capture_ep_freq: int = 1000
    """the frequency of capturing videos of the agent performances"""

    # Algorithm specific arguments
    env_id: str = "SamSegEnv-v0"
    """the id of the environment"""
    env_cfg_path: str = "configs/envs/repvit_sam_coco.yaml"
    """the environment configuration path"""
    agent_cfg_path: str = "configs/agents/explicit_agent.yaml"
    """the type of the agent"""
    total_timesteps: int = 20000000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-5
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

    # Evaluation and visualization
    eval_interval: int = 100
    """how often (in iterations) to run evaluation"""
    eval_episodes: int = 10
    """number of evaluation episodes each eval"""
    plot_interval: int = 50
    """how often (in iterations) to refresh and save matplotlib figures"""
    figures_dirname: str = "figures"
    """subdirectory under log_dir to store .png plots"""
    save_metrics_csv: bool = True
    """whether to also save a metrics.csv for post hoc plotting"""

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, capture_ep_freq, log_dir, env_cfg):
    def capture_ep(capture_ep_freq, episode_idx):
        return episode_idx % capture_ep_freq == 0

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, **env_cfg)
            video_folder = os.path.join(log_dir, "videos")
            env = gym.wrappers.RecordVideo(
                env, video_folder, episode_trigger=partial(capture_ep, capture_ep_freq)
            )
        else:
            env = gym.make(env_id, **env_cfg)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def load_obs_to_tensor(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    obs_tensor: Dict[str, torch.Tensor] = {}
    for key in obs.keys():
        if isinstance(obs[key], np.ndarray):
            obs_tensor[key] = torch.Tensor(obs[key]).to(device)
        else:
            obs_tensor[key] = obs[key]
    return obs_tensor


def update_obs_at_step(obs: Dict[str, torch.Tensor], next_obs: Dict[str, torch.Tensor], step: int) -> None:
    for key in obs.keys():
        obs[key][step] = next_obs[key]


def flatten_obs(obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_obs: Dict[str, torch.Tensor] = {}
    for key, val in obs.items():
        val_shape = list(val.shape)
        num_steps, num_envs = val_shape[:2]
        new_obs[key] = val.reshape(num_steps * num_envs, *val_shape[2:])
    return new_obs


def get_obs_at_inds(obs: Dict[str, torch.Tensor], mb_inds: np.ndarray) -> Dict[str, torch.Tensor]:
    new_obs: Dict[str, torch.Tensor] = {}
    for key, val in obs.items():
        new_obs[key] = val[mb_inds]
    return new_obs


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def save_metrics_csv(csv_path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    header = list(rows[0].keys())
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_training_curves(fig_path: str, metrics: List[Dict[str, float]]) -> None:
    if not metrics:
        return
    iters = [m["iteration"] for m in metrics]
    lr = [m["learning_rate"] for m in metrics]
    v_loss = [m["value_loss"] for m in metrics]
    p_loss = [m["policy_loss"] for m in metrics]
    entropy = [m["entropy"] for m in metrics]
    kl = [m["approx_kl"] for m in metrics]
    clipfrac = [m["clipfrac"] for m in metrics]
    ev = [m.get("explained_variance", 0.0) for m in metrics]
    epi = [m.get("mean_episodic_return", np.nan) for m in metrics]

    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(iters, v_loss, label="value_loss")
    ax1.plot(iters, p_loss, label="policy_loss")
    ax1.set_title("Losses")
    ax1.set_xlabel("iteration")
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(iters, entropy, label="entropy")
    ax2.set_title("Entropy")
    ax2.set_xlabel("iteration")

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(iters, kl, label="approx_kl")
    ax3.plot(iters, clipfrac, label="clipfrac")
    ax3.set_title("KL/Clipfrac")
    ax3.set_xlabel("iteration")
    ax3.legend()

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(iters, lr, label="learning_rate")
    ax4.plot(iters, ev, label="explained_var")
    if not all(np.isnan(epi)):
        ax4.plot(iters, epi, label="episodic_return")
    ax4.set_title("LR / ExplainedVar / Return")
    ax4.set_xlabel("iteration")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def plot_eval_curves(fig_path: str, eval_points: List[Tuple[int, float, float]]) -> None:
    if not eval_points:
        return
    iters = [p[0] for p in eval_points]
    dice = [p[1] for p in eval_points]
    iou = [p[2] for p in eval_points]

    plt.figure(figsize=(10, 4))
    plt.plot(iters, dice, label="mean_dice")
    plt.plot(iters, iou, label="mean_iou")
    plt.xlabel("iteration")
    plt.ylabel("score")
    plt.title("Validation Dice/IoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def compute_dice_iou(pred_mask: np.ndarray, gt_categorical_mask: np.ndarray, threshold: float = 0.5) -> Tuple[float, float]:
    # pred_mask: 2D float in [0,1] or logits already passed through sigmoid in env
    pred_bin = (pred_mask >= threshold).astype(np.uint8)
    # Merge instances into a single foreground mask, resize to pred size
    if gt_categorical_mask.ndim == 3:
        gt_any = np.any(gt_categorical_mask > 0, axis=-1).astype(np.uint8)
    else:
        gt_any = (gt_categorical_mask > 0).astype(np.uint8)
    if gt_any.shape != pred_bin.shape:
        gt_any = cv2.resize(gt_any, pred_bin.shape[::-1], interpolation=cv2.INTER_NEAREST)  # type: ignore

    intersection = np.logical_and(pred_bin == 1, gt_any == 1).sum()
    union = np.logical_or(pred_bin == 1, gt_any == 1).sum()
    pred_sum = pred_bin.sum()
    gt_sum = gt_any.sum()
    eps = 1e-6
    iou = float(intersection) / float(union + eps)
    dice = (2.0 * float(intersection)) / float(pred_sum + gt_sum + eps)
    return dice, iou


def to_tensor_batch_for_agent(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    batched: Dict[str, torch.Tensor] = {}
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            t = torch.from_numpy(val)
            if key == "sam_pred_mask_prob":
                t = t.unsqueeze(0)
            elif key == "sam_image_embeddings":
                t = t.unsqueeze(0)
            elif key == "image":
                t = t.unsqueeze(0)
            batched[key] = t.to(device)
    return batched


def evaluate_agent(agent: torch.nn.Module, env_cfg: Dict, device: torch.device, num_episodes: int = 10) -> Tuple[float, float]:
    # Lazy import to avoid mandatory dependency if user only trains
    import cv2  # noqa: F401

    from custom_gym_implns.envs.sam_seg_env import SamSegEnv

    env = SamSegEnv(**env_cfg)
    agent.eval()
    dice_scores: List[float] = []
    iou_scores: List[float] = []

    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _info = env.reset()
            max_steps = env.max_steps if env.max_steps is not None else 5
            for _ in range(max_steps):
                obs_batch = to_tensor_batch_for_agent(obs, device)
                action, _, _, _ = agent.get_action_and_value(obs_batch)
                action_int = int(action.item())
                obs, _reward, terminated, truncated, _info = env.step(action_int)
                if terminated or truncated:
                    break
            pred = env._sam_pred_mask
            gt = env._categorical_instance_masks
            d, j = compute_dice_iou(pred, gt)
            dice_scores.append(float(d))
            iou_scores.append(float(j))

    agent.train()
    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    return mean_dice, mean_iou


if __name__ == "__main__":
    import cv2  # used in compute_dice_iou resize

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
    figures_dir = os.path.join(log_dir, args.figures_dirname)
    ensure_dirs(log_dir, figures_dir)

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    ensure_dirs(checkpoint_dir)

    if (args.resume_from is not None) and (not os.path.exists(args.resume_from)):
        raise ValueError(f"resume_from {args.resume_from} does not exist")

    if args.accelerator:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id, i, args.capture_video, args.capture_ep_freq, log_dir, env_cfg
            )
            for i in range(args.num_envs)
        ],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = make_agent(agent_cfg, envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location="cpu")
        agent.load_state_dict(checkpoint["model"])
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Storage setup
    obs: Dict[str, torch.Tensor] = {}
    for key, space in envs.single_observation_space.spaces.items():
        if isinstance(space, gym.spaces.Box):
            obs[key] = torch.zeros((args.num_steps, args.num_envs) + space.shape).to(device)
        elif isinstance(space, gym.spaces.Text):
            obs[key] = np.array([""] * (args.num_steps * args.num_envs), dtype=object).reshape(
                (args.num_steps, args.num_envs)
            )
        elif isinstance(space, gym.spaces.Discrete):
            obs[key] = torch.zeros((args.num_steps, args.num_envs), dtype=torch.int64).to(device)
        else:
            raise NotImplementedError(f"Unsupported observation space type: {type(space)}")
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Metrics buffers
    metrics_rows: List[Dict[str, float]] = []
    eval_points: List[Tuple[int, float, float]] = []  # (iteration, mean_dice, mean_iou)
    eval_rows: List[Dict[str, float]] = []

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    raw_next_obs, _ = envs.reset(seed=args.seed)
    next_obs = load_obs_to_tensor(raw_next_obs, device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Collect episodic returns surfaced by RecordEpisodeStatistics
    episodic_returns_this_iter: List[float] = []

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - ((iteration - 1.0) / args.num_iterations)
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        episodic_returns_this_iter.clear()

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            update_obs_at_step(obs, next_obs, step)
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_raw_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.Tensor(reward).to(device).view(-1)
            next_obs = load_obs_to_tensor(next_raw_obs, device)
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_returns_this_iter.append(float(info["episode"]["r"]))

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = flatten_obs(obs)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs: List[float] = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = get_obs_at_inds(b_obs, mb_inds)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    mb_obs, b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
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

        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        print("Iteration:", iteration)

        # Aggregate metrics for this iteration
        mean_epi_ret = float(np.mean(episodic_returns_this_iter)) if episodic_returns_this_iter else np.nan
        row = {
            "iteration": iteration,
            "global_step": int(global_step),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "value_loss": float(v_loss.item()),
            "policy_loss": float(pg_loss.item()),
            "entropy": float(entropy_loss.item()),
            "old_approx_kl": float(old_approx_kl.item()),
            "approx_kl": float(approx_kl.item()),
            "clipfrac": float(np.mean(clipfracs) if clipfracs else 0.0),
            "explained_variance": float(explained_var if not np.isnan(explained_var) else 0.0),
            "sps": float(sps),
            "mean_episodic_return": mean_epi_ret,
        }
        metrics_rows.append(row)

        # Periodic evaluation
        if (iteration % max(1, args.eval_interval)) == 0:
            mean_dice, mean_iou = evaluate_agent(agent, env_cfg, device, num_episodes=args.eval_episodes)
            eval_points.append((iteration, mean_dice, mean_iou))
            eval_rows.append({"iteration": iteration, "mean_dice": float(mean_dice), "mean_iou": float(mean_iou)})
            print(f"[Eval] Iter {iteration}: mean Dice={mean_dice:.4f}, mean IoU={mean_iou:.4f}")

        # Plotting and CSV export
        if (iteration % max(1, args.plot_interval)) == 0:
            plot_training_curves(os.path.join(figures_dir, "training_curves.png"), metrics_rows)
            plot_eval_curves(os.path.join(figures_dir, "eval_curves.png"), eval_points)
            if args.save_metrics_csv:
                save_metrics_csv(os.path.join(log_dir, "metrics.csv"), metrics_rows)
                save_metrics_csv(os.path.join(log_dir, "eval_metrics.csv"), eval_rows)

        if iteration % args.checkpoint_iter_freq == 0:
            checkpoint = {
                "model": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pth"))
            latest_symbolic = os.path.join(checkpoint_dir, "latest.pth")
            try:
                if os.path.islink(latest_symbolic) or os.path.exists(latest_symbolic):
                    os.unlink(latest_symbolic)
                os.symlink(f"checkpoint_{iteration}.pth", latest_symbolic)
            except Exception:
                pass

    envs.close()

    # Final flush of figures/CSV
    plot_training_curves(os.path.join(figures_dir, "training_curves.png"), metrics_rows)
    plot_eval_curves(os.path.join(figures_dir, "eval_curves.png"), eval_points)
    if args.save_metrics_csv:
        save_metrics_csv(os.path.join(log_dir, "metrics.csv"), metrics_rows)
        save_metrics_csv(os.path.join(log_dir, "eval_metrics.csv"), eval_rows)

    print(f"Saved figures to: {figures_dir}")
    print(f"Saved metrics CSV to: {os.path.join(log_dir, 'metrics.csv')}")


