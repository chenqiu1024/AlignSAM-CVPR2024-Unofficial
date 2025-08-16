import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from torchvision.transforms.functional import to_pil_image

from CLIP_Surgery import clip
from CLIP_Surgery.clip.clip_model import ResidualAttentionBlock


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ExplicitAgent(nn.Module):
    def __init__(self, envs, agent_cfg):
        super().__init__()
        # Explicit agent fuses two streams:
        # 1) SAM image encoder features and current mask belief (prompt-conditioned vision features)
        # 2) CLIP-surgery similarity maps to target text prompts (language grounding)
        # Fusion is followed by a lightweight attention block and an actor-critic head for PPO.
        #
        # References:
        # - Segment Anything: promptable segmentation via point prompts (Kirillov et al., 2023)
        #   docs/Segment Anything.pdf
        # - AlignSAM: aligning SAM to open context via RL; PPO agent optimizes interactive prompts
        #   to maximize segmentation reward
        #   docs/AlignSAM- Aligning Segment Anything Model to Open Context via Reinforcement Learning.pdf
        # - CLIP Surgery: removes dataset bias and provides class activation-like similarity maps
        #   for better localization
        #   CLIP_Surgery repo (see CLIP_Surgery/README.md in this project subtree)

        self.setup_clip(
            agent_cfg['clip_model_name'], 
            agent_cfg['clip_image_size'],
            agent_cfg['clip_text_prompt']
        )

        self.sam_network = nn.Sequential(
            layer_init(nn.Conv2d(256, 128, 3, stride=2, padding=1, padding_mode='zeros')), # (b, 256, 64, 64)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 64, 3, stride=2, padding=1, padding_mode='zeros')), # (b, 128, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 5, stride=3)), # (b, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(), # (b, 1024)
        )
        # The SAM stream conditions on current mask probability (see get_sam_features) to focus the
        # encoder features on regions consistent with past prompts, akin to iterative refinement.

        self.clip_network = nn.Sequential(
            layer_init(nn.Conv2d(len(self.clip_text_prompt), 16, 3, stride=2, padding=1, padding_mode='zeros')), # (b, num_prompts, 64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(),  
            layer_init(nn.Conv2d(16, 32, 3, stride=2, padding=1, padding_mode='zeros')), # (b, 16, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 5, stride=3)), # (b, 64, 4, 4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(), # (b, 1024)
        )
        # The CLIP-surgery stream converts text-image similarity to spatial maps (class-activation-like)
        # to inject target category semantics; see get_clip_surgery_features.

        self.combined_attention = ResidualAttentionBlock(
            d_model=2048,
            n_head=4,
            attn_mask=None,
            need_weights=False
        ) 
        # A shallow attention block enables cross-stream interaction prior to policy/value heads.

        self.head = nn.Sequential(
            layer_init(nn.Linear(2048, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)


    def setup_clip(self, clip_model_name, clip_image_size, clip_text_prompt):
        clip_model, _ = clip.load(clip_model_name, device='cpu')
        self.clip_model = clip_model
        self.clip_model.eval()

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.clip_preprocess = Compose([
            Resize((clip_image_size[1], clip_image_size[0]), interpolation=InterpolationMode.BICUBIC), 
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        self.clip_text_prompt = clip_text_prompt

        # Prompt ensemble for text features with normalization
        text_features = clip.encode_text_with_prompt_ensemble(self.clip_model, clip_text_prompt, 'cpu')
        self.clip_text_features = nn.Parameter(text_features, requires_grad=False)


        # Extract redundant features from an empty string
        redundant_features = clip.encode_text_with_prompt_ensemble(self.clip_model, [""], 'cpu')
        self.clip_redundant_features = nn.Parameter(redundant_features, requires_grad=False)
        

    def get_clip_surgery_features(self, obs):
        with torch.no_grad():
            obs_image = obs["image"] # (b, h, w, c)
            embedding_shape = tuple(obs["sam_image_embeddings"].size()) # (b, c, h, w)
            
            image = obs_image.float().permute(0, 3, 1, 2) / 255.0 # (b, h, w, c) -> (b, c, h, w)
            image = self.clip_preprocess(image)

            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            # Apply feature surgery (see CLIP Surgery method): remove redundant components to obtain
            # sharper text-conditioned localization maps for the target categories.
            similarity = clip.clip_feature_surgery(image_features, 
                                                self.clip_text_features, 
                                                self.clip_redundant_features)
            similarity_map = clip.get_similarity_map(similarity[:, 1:, :], 
                                                    (embedding_shape[2], embedding_shape[3]))
            
            similarity_map = similarity_map.permute(0, 3, 1, 2) # (b, h, w, c) -> (b, c, h, w)

            return similarity_map


    def get_sam_features(self, obs): ###!!!HERE
        sam_image_embeddings = obs["sam_image_embeddings"] # (b, c, h, w)
        sam_pred_mask_prob = obs["sam_pred_mask_prob"].unsqueeze(dim=1)  # (b, 1, h, w)

        embedding_shape = tuple(sam_image_embeddings.size()) # (b, c, h, w)
        resized_sam_mask_prob = nn.functional.interpolate(
            sam_pred_mask_prob, size=(embedding_shape[2], embedding_shape[3]), 
            mode="bilinear", align_corners=False)
        resized_sam_mask_prob = resized_sam_mask_prob.repeat(1, embedding_shape[1], 1, 1) # Duplicate in Channel to align with 

        x = sam_image_embeddings * resized_sam_mask_prob 
        x += sam_image_embeddings # skip connection

        return x
    

    def merge_clip_sam_features(self, obs):
        x_sam = self.get_sam_features(obs)
        x_clip = self.get_clip_surgery_features(obs)
        hidden_x = self.sam_network(x_sam)
        hidden_clip = self.clip_network(x_clip)
        combined_hidden = torch.cat([hidden_x, hidden_clip], dim=1)
        out = self.combined_attention(combined_hidden)
        return out
        # The merged representation is used by the actor-critic head to decide the next prompt.
        # This implements the AlignSAM idea of leveraging both vision and language cues during RL.


    def get_value(self, obs):
        x = self.merge_clip_sam_features(obs)
        hidden = self.head(x)
        return self.critic(hidden)


    def get_action_and_value(self, obs, action=None):
        x = self.merge_clip_sam_features(obs)
        hidden = self.head(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
