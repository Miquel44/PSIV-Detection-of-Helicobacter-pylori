import torch
import torch.nn as nn

from Sistema2.Attention.AttentionUnits import GatedAttention, NeuralNetwork
from Sistema2.Triplets.training_triplet import PatchEmbeddingFC, LATENT_DIM, EMBED_DIM

class PatientAttentionModel(nn.Module):
    """
    Modelo MIL a nivel paciente:
    - Para cada patch: latent_features (262144) -> PatchEmbeddingFC (512)
    - Atención sobre los parches -> vector de paciente
    - MLP final -> logits de paciente (2 clases)
    """
    def __init__(self, in_dim_latent=LATENT_DIM, embed_dim=EMBED_DIM,
                 attention_branches=1, n_classes=2):
        super().__init__()

        # Embedding de patch (preentrenado con Triplet)
        self.embed = PatchEmbeddingFC(in_dim=in_dim_latent, embed_dim=embed_dim)

        # Atención
        att_params = {
            'in_features': embed_dim,
            'decom_space': embed_dim // 2,   # por ejemplo 256
            'ATTENTION_BRANCHES': attention_branches
        }
        self.attention = GatedAttention(att_params)

        # MLP final a nivel paciente
        mlp_in = embed_dim * attention_branches
        net_params = {
            'in_features': mlp_in,
            'out_features': n_classes
        }
        self.classifier = NeuralNetwork(net_params)

    def forward(self, patch_latents):
        """
        patch_latents: tensor (N_patches, LATENT_DIM) para UN paciente.
        """
        # 1) Embedding de cada patch
        e = self.embed(patch_latents)           # (N_patches, 512)

        # 2) Atención (espera input con batch dim 1 x NV x M)
        H = e.unsqueeze(0)                      # (1, N_patches, 512)
        Z, A = self.attention(H)               # Z: (ATT_BRANCHES, 512)

        # 3) Vector de paciente + MLP
        Z_flat = Z.view(1, -1)                 # (1, ATT_BRANCHES*512)
        logits = self.classifier(Z_flat)       # (1, 2)

        return logits, A