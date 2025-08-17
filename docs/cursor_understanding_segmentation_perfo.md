# Understanding segmentation performance in AlignSAM
_Exported on 8/12/2025 at 21:21:48 PDT from Cursor (1.4.3)_

---

**User**

Given train_sam_align_ppo.py as the training code of AlignSAM, figure out where does it get the segmentation performance(such as DICE/IOU values) for each sample input. Do not change any existing code, just try hard to understand the whole project and research on the problem.

---

**Cursor**

I’m going to scan the training script and related environment/dataset/model code to locate where per-sample segmentation metrics (IoU/Dice) are computed, and how they’re used (e.g., as PPO rewards). I’ll search broadly for IoU/Dice and reward logic across the repo, then drill into the specific files that look relevant.

Scanning the RL environment and dataset modules to locate where Dice/IoU are computed and how the env returns or logs them. I'll open the SAM env, its RepViT-SAM wrapper, and dataset loader, plus the env package init and env config.

### Short answer
- The per-sample segmentation quality is computed inside the gym environment, not in the training loop. Specifically:
  - Ground-truth Dice is computed in `custom_gym_implns/envs/sam_seg_env.py` in `compute_reward(...)` by comparing SAM’s predicted mask to the dataset’s ground-truth mask.
  - IoU values used during inference in the environment come from SAM’s internal IoU prediction head (predicted IoU), returned by the RepViT-SAM `SamPredictor.predict(...)`. These are used to pick the best of multiple candidate masks; they are not GT IoU.

### Code pointers

- Dice computation against GT in the environment (used for reward shaping, tracks `_last_score` and `dice_reward`):
```187:231:custom_gym_implns/envs/sam_seg_env.py
    def compute_reward(self, pred_mask, act):
        # Compute dice score
        # Reward shaping:
        # - Dice improvement and correctness of click semantics.
        # - Encourages progressive refinement as in AlignSAM: higher reward when the mask improves
        #   and when the click matches ground-truth foreground/background.
        resized_gt_mask = cv2.resize(self._gt_mask,
                                     pred_mask.shape[::-1],
                                     interpolation=cv2.INTER_NEAREST)
        assert resized_gt_mask.shape == pred_mask.shape

        intersection = np.sum(resized_gt_mask * pred_mask)
        union = np.sum(resized_gt_mask) + np.sum(pred_mask) #- intersection

        eps = 1e-6
        dice_score = (2 * intersection + eps)/ (union + eps)
        dice_reward = dice_score - self._last_score
        self._last_score = max(dice_score, self._last_score)
        ...
        # reward = dice_reward + correct_input_reward
        reward = correct_input_reward
        # Add bonus reward if dice score is above a threshold and not the first action
        reward += int((dice_reward > 0.05) and len(self._last_actions)> 0)* 2
        return reward
```

- Predicted IoU from SAM used to select the best candidate mask in the env:
```161:173:custom_gym_implns/envs/sam_seg_env.py
    def run_sam(self):
        # Convert accumulated clicks to masks via SAM predictor. This follows the promptable inference
        # pipeline of SAM (Kirillov et al., 2023), selecting the mask with highest IoU score.
        masks, ious, low_res_mask_logits = self.sam_predictor.predict(
            np.array(self._last_actions["input_points"]), 
            np.array(self._last_actions["input_labels"])
        )

        best_mask_idx = np.argmax(ious)
        best_pred_mask_prob = 1 / (1 + np.exp(-low_res_mask_logits[best_mask_idx]))
        self._sam_pred_mask_prob = best_pred_mask_prob
        self._sam_pred_mask = (1 / (1 + np.exp(-masks[best_mask_idx]))).astype(np.float32)
```

- RepViT-SAM wrapper showing the `predict(...)` API that returns masks, predicted IoU, and logits:
```56:62:custom_gym_implns/envs/utils/repvit_sam_wrapper.py
        masks, ious, low_res_mask_logits = self.sam_predictor.predict(
            input_points, input_labels,
            multimask_output=self.multimaskoutput,
            return_logits=True
        )

        return masks, ious, low_res_mask_logits
```

- Ground-truth mask source (dataset sample used by the env):
```44:61:datasets/coco_dataset.py
    def get_sample(self, target_categories):
        target_cat_ids, img_ids = self.configure_targets(target_categories)

        while True:
            sample_img_id = self.random.choice(img_ids)
            sample_img = self.load_image(sample_img_id)
            ...
            ann_ids = self.coco.getAnnIds(imgIds=sample_img_id, 
                                        catIds=target_cat_ids, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)

            anns = sorted(anns, key=lambda x: x['area'], reverse=True)
            sample_mask = self.coco.annToMask(anns[0])
            ...
        return sample_img, sample_mask
```

### Clarifications
- The training loop in `train_sam_align_ppo.py` never computes IoU/Dice itself; it only consumes `reward` from the env.
- IoU in `run_sam()` is the model’s predicted IoU (mask quality proxy) used to choose a mask. True IoU against GT is not computed; only Dice is measured against GT inside `compute_reward(...)`.

- The env’s `info` includes `sam_pred_mask` and click history but does not expose Dice/IoU explicitly; Dice is internal to reward shaping.

- If you need per-step GT IoU logs, they’re not present; only the Dice calculation exists internally in `compute_reward(...)`.

- Summary
  - Dice against GT is computed in `custom_gym_implns/envs/sam_seg_env.py:compute_reward`, using the GT mask from `datasets/coco_dataset.py:get_sample`.
  - IoU used for mask selection is SAM’s predicted IoU from `RepVITSamWrapper.predict` and is not a GT IoU.
  - The training script does not directly compute or log per-sample IoU/Dice; it uses the env’s reward.

```python
    def compute_reward(self, pred_mask, act):
        # Compute dice score
        # Reward shaping:
        # - Dice improvement and correctness of click semantics.
        # - Encourages progressive refinement as in AlignSAM: higher reward when the mask improves
        #   and when the click matches ground-truth foreground/background.
        resized_gt_mask = cv2.resize(self._gt_mask,
                                     pred_mask.shape[::-1],
                                     interpolation=cv2.INTER_NEAREST)
        assert resized_gt_mask.shape == pred_mask.shape

        intersection = np.sum(resized_gt_mask * pred_mask)
        union = np.sum(resized_gt_mask) + np.sum(pred_mask) #- intersection

        eps = 1e-6
        dice_score = (2 * intersection + eps)/ (union + eps)
        dice_reward = dice_score - self._last_score
        self._last_score = max(dice_score, self._last_score)
        ...
        # reward = dice_reward + correct_input_reward
        reward = correct_input_reward
        # Add bonus reward if dice score is above a threshold and not the first action
        reward += int((dice_reward > 0.05) and len(self._last_actions)> 0)* 2
        return reward
```

```python
    def run_sam(self):
        # Convert accumulated clicks to masks via SAM predictor. This follows the promptable inference
        # pipeline of SAM (Kirillov et al., 2023), selecting the mask with highest IoU score.
        masks, ious, low_res_mask_logits = self.sam_predictor.predict(
            np.array(self._last_actions["input_points"]), 
            np.array(self._last_actions["input_labels"])
        )

        best_mask_idx = np.argmax(ious)
        best_pred_mask_prob = 1 / (1 + np.exp(-low_res_mask_logits[best_mask_idx]))
        self._sam_pred_mask_prob = best_pred_mask_prob
        self._sam_pred_mask = (1 / (1 + np.exp(-masks[best_mask_idx]))).astype(np.float32)
```

```python
        masks, ious, low_res_mask_logits = self.sam_predictor.predict(
            input_points, input_labels,
            multimask_output=self.multimaskoutput,
            return_logits=True
        )

        return masks, ious, low_res_mask_logits
```

```python
    def get_sample(self, target_categories):
        target_cat_ids, img_ids = self.configure_targets(target_categories)

        while True:
            sample_img_id = self.random.choice(img_ids)
            sample_img = self.load_image(sample_img_id)
            ...
            ann_ids = self.coco.getAnnIds(imgIds=sample_img_id, 
                                        catIds=target_cat_ids, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)

            anns = sorted(anns, key=lambda x: x['area'], reverse=True)
            sample_mask = self.coco.annToMask(anns[0])
            ...
        return sample_img, sample_mask
```

