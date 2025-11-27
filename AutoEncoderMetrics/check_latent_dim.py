import torch
from Models.AEmodels import Encoder

if __name__ == "__main__":
    # mismos parámetros que usas en AEExample_Script para Config '1'
    inputmodule_paramsEnc = {
        'num_input_channels': 3,
    }
    net_paramsEnc = {
        'block_configs': [[32, 32], [64, 64]],
        'stride': [[1, 2], [1, 2]],
    }

    enc = Encoder(inputmodule_paramsEnc, net_paramsEnc)
    enc.eval()

    # dummy 256x256
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        z = enc(x)

    print("Encoder output shape:", z.shape)
    print("Latent dim (flattened):", z.view(1, -1).shape[1])