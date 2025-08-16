import os
import time
import argparse
import yaml
import numpy as np
import cv2
import torch

from typing import Dict, Any, Tuple

# Use project modules
from custom_gym_implns.envs.sam_seg_env import SamSegEnv
from models import make_agent


class SingleEnvSpec:
    """
    Minimal adapter to satisfy agent construction which expects `.single_action_space.n`.
    """
    def __init__(self, action_space):
        class _AS:
            def __init__(self, n):
                self.n = n
        self.single_action_space = _AS(action_space.n)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def to_tensor_batch(obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    # Create 1-batch tensors on CPU to match agent codepaths
    batched: Dict[str, torch.Tensor] = {}
    for key, val in obs.items():
        if key == "sam_pred_mask_prob":
            # shape (H, W) -> (1, H, W)
            t = torch.from_numpy(val).unsqueeze(0)
        elif key == "sam_image_embeddings":
            # shape (C, H, W) -> (1, C, H, W)
            t = torch.from_numpy(val).unsqueeze(0)
        elif key == "image":
            # shape (H, W, C) -> (1, H, W, C)
            t = torch.from_numpy(val).unsqueeze(0)
        else:
            continue
        batched[key] = t
    return batched


def compute_dice_iou(pred_mask_prob: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> Tuple[float, float]:
    assert pred_mask_prob.shape == gt_mask.shape, "Prediction and GT masks must have the same shape"
    pred_bin = (pred_mask_prob >= threshold).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_bin == 1, gt_bin == 1).sum()
    union = np.logical_or(pred_bin == 1, gt_bin == 1).sum()
    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()

    eps = 1e-6
    iou = float(intersection) / float(union + eps)
    dice = (2.0 * float(intersection)) / float(pred_sum + gt_sum + eps)
    return dice, iou


def overlay_masks(image_rgb: np.ndarray, pred_mask_prob: np.ndarray, gt_mask: np.ndarray, points: list, labels: list) -> np.ndarray:
    h, w, _ = image_rgb.shape
    pred_bin = (pred_mask_prob >= 0.5).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    overlay = image_rgb.copy()

    # Red for GT, Green for prediction
    gt_color = np.zeros_like(overlay)
    gt_color[:, :, 2] = (gt_bin * 255)
    pred_color = np.zeros_like(overlay)
    pred_color[:, :, 1] = (pred_bin * 255)

    alpha = 0.35
    overlay = cv2.addWeighted(overlay, 1.0, gt_color, alpha, 0)
    overlay = cv2.addWeighted(overlay, 1.0, pred_color, alpha, 0)

    # Draw clicks: green=positive(1), red=negative(0)
    for point, label in zip(points, labels):
        x, y = int(point[0]), int(point[1])
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.circle(overlay, (x, y), 6, color, -1)

    return overlay


def run_inference(args: argparse.Namespace) -> None:
    env_cfg = load_yaml(args.env_cfg)
    agent_cfg = load_yaml(args.agent_cfg)

    os.makedirs(args.output_dir, exist_ok=True)

    # Instantiate environment directly from config
    env = SamSegEnv(**env_cfg)

    # Build agent and load trained weights
    spec = SingleEnvSpec(env.action_space)
    agent = make_agent(agent_cfg, spec)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu")
    agent.load_state_dict(state["model"])  # training script saved under key 'model'
    agent.eval()

    # Evaluation loop
    num_episodes = int(args.num_episodes)
    num_visualize = int(min(args.num_visualize, num_episodes))

    dice_scores = []
    iou_scores = []

    for episode_idx in range(num_episodes):
        obs, info = env.reset()

        # Roll out for max_steps (env does not support 'done' action in current config)
        for step_idx in range(env.max_steps if env.max_steps is not None else 5):
            obs_batch = to_tensor_batch(obs)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_batch)
            action_int = int(action.item())

            obs, reward, terminated, truncated, info = env.step(action_int)
            if terminated or truncated:
                break

        # Final prediction and metrics
        pred_mask_prob = env._sam_pred_mask  # probability map in [0,1]
        gt_mask = env._gt_mask  # binary mask 0/1
        image_rgb = env._image  # RGB image (H,W,C)

        dice, iou = compute_dice_iou(pred_mask_prob, gt_mask, threshold=args.threshold)
        dice_scores.append(dice)
        iou_scores.append(iou)

        # Visualization
        if episode_idx < num_visualize:
            overlay = overlay_masks(
                image_rgb=image_rgb,
                pred_mask_prob=pred_mask_prob,
                gt_mask=gt_mask,
                points=env._last_actions["input_points"],
                labels=env._last_actions["input_labels"],
            )

            out_path = os.path.join(args.output_dir, f"episode_{episode_idx:04d}_dice_{dice:.3f}_iou_{iou:.3f}.png")
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"Episode {episode_idx+1}/{num_episodes}: Dice={dice:.4f}, IoU={iou:.4f}")

    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    print("-" * 60)
    print(f"Mean Dice over {num_episodes} episodes: {mean_dice:.4f}")
    print(f"Mean IoU  over {num_episodes} episodes: {mean_iou:.4f}")


def build_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference and evaluation for AlignSAM trained agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint .pth")
    parser.add_argument("--env-cfg", type=str, default="configs/envs/repvit_sam_coco.yaml", help="Path to env YAML config")
    parser.add_argument("--agent-cfg", type=str, default="configs/agents/explicit_agent.yaml", help="Path to agent YAML config")
    parser.add_argument("--num-episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--num-visualize", type=int, default=10, help="Number of episodes to save visualizations for")
    ts = int(time.time())
    parser.add_argument("--output-dir", type=str, default=os.path.join("logs", f"inference_{ts}"), help="Directory to save outputs")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binarizing predicted mask")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_argparser()
    run_inference(args)


