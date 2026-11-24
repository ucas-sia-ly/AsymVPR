# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch

class DinoV2(torch.nn.Module):
    AVAILABLE_MODELS = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vitg14'
    ]
    
    def __init__(
        self,
        backbone_name="dinov2_vitb14",
        unfreeze_n_blocks=2,
        reshape_output=True,
        dino=None,
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.reshape_output = reshape_output
        
        if backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(f'Unknown DINOv2 backbone: {backbone_name}')
        self.dino = dino if dino is not None else torch.hub.load(
            'facebookresearch/dinov2', backbone_name)
        if type(unfreeze_n_blocks) is not int or not 0 <= unfreeze_n_blocks <= len(self.dino.blocks):
            raise ValueError('unfreeze_n_blocks must be between zero and the number of DINOv2 blocks')
        self.dino.requires_grad_(False)
        if unfreeze_n_blocks:
            for block in self.dino.blocks[-unfreeze_n_blocks:]:
                block.requires_grad_(True)

        self.out_channels = self.dino.embed_dim
        
    @property
    def patch_size(self):
        return self.dino.patch_embed.patch_size[0]  # Assuming square patches
    
    def forward(self, x):
        B, _, H, W = x.shape
        frozen_blocks = len(self.dino.blocks) - self.unfreeze_n_blocks
        # No need to compute gradients for frozen layers
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)
            for blk in self.dino.blocks[:frozen_blocks]:
                x = blk(x)

        # Last blocks are trained
        for blk in self.dino.blocks[frozen_blocks:]:
            x = blk(x)
            
        
        x = x[:, 1:] # remove the [CLS] token
        
        # reshape the output tensor to B, C, H, W
        if self.reshape_output:
            _, _, C = x.shape # or C = self.embed_dim
            patch_size = self.patch_size
            x = x.permute(0, 2, 1).view(B, C, H // patch_size, W // patch_size)
        return x
