"""BoQ query model and Eq. (8); does not alter the original BoQ training code."""
import torch
from torch import nn
from torch.nn import functional as F

from src.boq import BoQ


class EfficientViTBoQ(nn.Module):
    def __init__(self, config, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            config['student_name'], pretrained=pretrained,
            features_only=True, out_indices=(3,),
        )
        self.aggregator = BoQ(
            in_channels=self.backbone.feature_info.channels()[-1],
            proj_channels=config['proj_channels'], num_queries=config['num_queries'],
            num_layers=config['num_layers'],
            row_dim=config['descriptor_dim'] // config['proj_channels'],
        )

    def forward(self, images):
        return self.aggregator(self.backbone(images)[-1])[0]


class GalleryBoQ(nn.Module):
    def __init__(self, config):
        super().__init__()
        from src.backbones import DinoV2
        # Construct without a second network download; load the complete VPR state below.
        dino = torch.hub.load(config['dino_source'], 'dinov2_vitb14',
                              source='local', pretrained=False)
        backbone = DinoV2.__new__(DinoV2)
        nn.Module.__init__(backbone)
        backbone.dino = dino
        backbone.backbone_name = 'dinov2_vitb14'
        backbone.unfreeze_n_blocks = 2
        backbone.reshape_output = True
        backbone.out_channels = dino.embed_dim
        self.backbone = backbone
        self.aggregator = BoQ(in_channels=768, proj_channels=384,
                              num_queries=64, num_layers=2, row_dim=32)
        self.load_state_dict(torch.load(config['teacher_weights'],
                                        map_location='cpu', weights_only=True), strict=True)
        self.requires_grad_(False)
        self.eval()

    def forward(self, images):
        return self.aggregator(self.backbone(images))[0]


def asym_loss(q, g, place_ids, centroids, variances, temperature=0.05,
              gamma=15., chunk_size=4096):
    """Eq. (8), all other places as negatives, unnormalised arithmetic centroids.

    Chunk the fixed bank to bound temporary logits. Autograd saves the chunks
    and computes the exact logsumexp over the complete geographical bank.
    """
    positive = (q * g).sum(-1) / temperature
    regularizer = gamma / (2 * temperature ** 2) * (q.square() * variances[place_ids]).sum(-1)
    negative_lse = torch.full_like(positive, -torch.inf)
    for start in range(0, len(centroids), chunk_size):
        stop = min(start + chunk_size, len(centroids))
        logits = q @ centroids[start:stop].T / temperature
        own = (place_ids >= start) & (place_ids < stop)
        rows = own.nonzero(as_tuple=True)[0]
        logits[rows, place_ids[rows] - start] = -torch.inf
        negative_lse = torch.logaddexp(negative_lse, torch.logsumexp(logits, dim=-1))
    return F.softplus(negative_lse + regularizer - positive).mean()
