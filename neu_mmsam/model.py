import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from typing import Optional, Dict, Union, List, Tuple
import numpy as np

from segment_anything import sam_model_registry
from segment_anything.modeling.image_encoder import PatchEmbed

# -- Simplified utility functions (originally from utilbox) --

def fix_params(model: nn.Module):
    """Freezes the parameters of a model."""
    for param in model.parameters():
        param.requires_grad = False

def data_instance_norm(
    data: Union[torch.Tensor, List[torch.Tensor]],
    norm_type: Optional[str] = 'mean-std'
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict]:
    """Normalizes a tensor or a list of tensors."""
    if norm_type is None:
        return data, {}

    if isinstance(data, torch.Tensor):
        if len(data.shape) == 3: # (C, H, W)
            data = data.unsqueeze(0) # (B, C, H, W)

        if norm_type == 'min-max':
            min_val = torch.min(data.view(data.size(0), -1), dim=1)[0]
            max_val = torch.max(data.view(data.size(0), -1), dim=1)[0]
            range_val = max_val - min_val
            range_val[range_val == 0] = 1.0
            normalized = (data - min_val[:, None, None, None]) / range_val[:, None, None, None]
        elif norm_type == 'mean-std':
            mean = torch.mean(data.view(data.size(0), -1), dim=1)
            std = torch.std(data.view(data.size(0), -1), dim=1)
            std[std == 0] = 1.0
            normalized = (data - mean[:, None, None, None]) / std[:, None, None, None]
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
        return normalized, {}

    elif isinstance(data, list):
        normalized_list = []
        for item in data:
            normalized_item, _ = data_instance_norm(item, norm_type)
            normalized_list.append(normalized_item)
        return normalized_list, {}
    else:
        raise TypeError(f"Unsupported data type for normalization: {type(data)}")


# -- Base Wrapper Classes --

class BaseImgEncoderWrapper(nn.Module):
    def __init__(self, ori_img_encoder: nn.Module, fix: bool = True):
        super(BaseImgEncoderWrapper, self).__init__()
        self.sam_img_encoder = ori_img_encoder
        if fix:
            fix_params(self.sam_img_encoder)

    def set_patch_embed(self, new_patch_embed: torch.nn.Module):
        self.sam_img_encoder.patch_embed = new_patch_embed

    @property
    def patch_embed(self):
        return self.sam_img_encoder.patch_embed

    def forward(self, x):
        return self.sam_img_encoder(x)

class BasePromptEncodeWrapper(nn.Module):
    def __init__(self, ori_prompt_encoder: nn.Module, fix: bool = True):
        super(BasePromptEncodeWrapper, self).__init__()
        self.sam_prompt_encoder = ori_prompt_encoder
        if fix:
            fix_params(self.sam_prompt_encoder)

    def forward(self, points=None, boxes=None, masks=None):
        return self.sam_prompt_encoder(points, boxes, masks)

    def get_dense_pe(self):
        return self.sam_prompt_encoder.get_dense_pe()

class BaseMaskDecoderWrapper(nn.Module):
    def __init__(self, ori_mask_decoder: nn.Module, fix: bool = True):
        super(BaseMaskDecoderWrapper, self).__init__()
        self.sam_mask_decoder = ori_mask_decoder
        if fix:
            fix_params(self.sam_mask_decoder)

    def forward(self, image_embeddings, prompt_encoder, sparse_embeddings, dense_embeddings, multimask_output=True):
        low_res_masks, iou_predictions = self.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        return low_res_masks, iou_predictions

# -- LoRA Encoder for Depth Data --

class XLoraEncoder(Module):
    def __init__(
            self,
            x_channel_num: int,
            rgb_encoder: Module,
            lora_rank: int = 4, # lora_rank is unused in this simplified version but kept for compatibility
            norm_type: Optional[str] = 'mean-std',
    ):
        super().__init__()
        self.norm_type = norm_type

        # In this simplified MVP, we create a copy of the original SAM image encoder
        # and replace its first convolution layer (patch_embed) to accept a different
        # number of input channels (1 for depth). The fine-tuning of this new
        # encoder is handled by the training script, which will only pass this
        # encoder's parameters to the optimizer.

        self.encoder = copy.deepcopy(rgb_encoder.sam_img_encoder)

        # Get properties from the original patch embed layer to build the new one
        orig_patch_embed = self.encoder.patch_embed
        embed_dim = orig_patch_embed.proj.out_channels
        patch_size = orig_patch_embed.proj.kernel_size

        # Create and set the new patch embed layer for the depth input
        self.encoder.patch_embed = PatchEmbed(
            kernel_size=patch_size,
            stride=patch_size,
            in_chans=x_channel_num,
            embed_dim=embed_dim,
        )

    def preprocess(self, img: Union[torch.Tensor, List[torch.Tensor]]):
        x_images, _ = data_instance_norm(img, self.norm_type)
        return x_images

    def forward(self, x_images: torch.Tensor):
        return self.encoder(x_images)

    @property
    def device(self):
        return self.encoder.patch_embed.proj.weight.device

# -- Main Model: SAM for UCMT --

import os

class SAMbyUCMT(nn.Module):
    def __init__(
            self,
            model_type: str = 'vit_b',
            sam_checkpoint: Optional[str] = None,
            x_channel_num: int = 1,
            x_lora_rank: int = 4,
            fix_img_en: bool = True,
            fix_prompt_en: bool = True,
            fix_mask_de: bool = True,
            multimask_output: bool = False
    ):
        super().__init__()

        # If checkpoint is provided but doesn't exist, proceed with random weights.
        if sam_checkpoint and not os.path.exists(sam_checkpoint):
            print(f"WARNING: SAM checkpoint not found at '{sam_checkpoint}'. The model will initialize with random weights.")
            sam_checkpoint = None

        # Build the original SAM model
        ori_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

        # Build the original SAM model
        ori_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

        self.sam_img_size = (ori_sam.image_encoder.img_size, ori_sam.image_encoder.img_size)
        self.register_buffer("pixel_mean", ori_sam.pixel_mean, False)
        self.register_buffer("pixel_std", ori_sam.pixel_std, False)
        self.mask_threshold = ori_sam.mask_threshold
        self.multimask_output = multimask_output

        # Wrap SAM components
        self.image_encoder_wrapper = BaseImgEncoderWrapper(ori_sam.image_encoder, fix=fix_img_en)
        self.prompt_encoder = BasePromptEncodeWrapper(ori_sam.prompt_encoder, fix=fix_prompt_en)
        self.mask_decoder = BaseMaskDecoderWrapper(ori_sam.mask_decoder, fix=fix_mask_de)

        # Create the depth encoder
        self.depth_encoder = XLoraEncoder(
            x_channel_num=x_channel_num,
            lora_rank=x_lora_rank,
            rgb_encoder=self.image_encoder_wrapper
        )

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, depth_images, points=None, boxes=None, masks=None):
        # Preprocess and encode depth images
        depth_images_norm = self.depth_encoder.preprocess(depth_images)
        image_embeddings = self.depth_encoder(depth_images_norm)

        # The original SAM implementation processes images one by one.
        # We loop through the batch to maintain compatibility.
        outputs_masks = []
        outputs_iou_preds = []
        for i in range(image_embeddings.shape[0]):
            single_image_embedding = image_embeddings[i:i+1]

            # For this MVP, we only use box prompts during training
            single_boxes = boxes[i:i+1] if boxes is not None else None

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=single_boxes,
                masks=None,
            )

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=single_image_embedding,
                prompt_encoder=self.prompt_encoder,
                sparse_embeddings=sparse_embeddings,
                dense_embeddings=dense_embeddings,
                multimask_output=self.multimask_output,
            )

            outputs_masks.append(low_res_masks)
            outputs_iou_preds.append(iou_predictions)

        all_masks = torch.cat(outputs_masks, dim=0)
        all_iou_preds = torch.cat(outputs_iou_preds, dim=0)

        # Postprocess masks to original image size
        ori_res_masks = F.interpolate(
            all_masks,
            size=(depth_images.shape[-2], depth_images.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        return ori_res_masks, all_iou_preds

    def infer(self, depth_image):
        """A simplified inference method for a single image without prompts."""

        # Preprocess and encode the depth image
        depth_image_norm = self.depth_encoder.preprocess(depth_image.to(self.device))

        # The model expects a batch, so we add a batch dimension
        if len(depth_image_norm.shape) == 3:
            depth_image_norm = depth_image_norm.unsqueeze(0)

        image_embeddings = self.depth_encoder(depth_image_norm)

        # Create a dummy prompt (a single point in the center)
        # This is required by the mask decoder, but for semantic segmentation,
        # we can often get a reasonable result with a generic prompt.
        batch_size = image_embeddings.shape[0]
        points = (
            torch.tensor([[[512, 512]]], device=self.device).float().expand(batch_size, -1, -1),
            torch.tensor([[1]], device=self.device).float().expand(batch_size, -1),
        )

        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=points, boxes=None, masks=None)

        # Decode masks
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            prompt_encoder=self.prompt_encoder,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Postprocess to original image size
        original_size = (depth_image.shape[-2], depth_image.shape[-1])
        upscaled_masks = F.interpolate(
            low_res_masks,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )

        # Apply threshold to get binary mask
        binary_mask = (upscaled_masks > self.mask_threshold).squeeze(1).to(torch.uint8)

        return binary_mask