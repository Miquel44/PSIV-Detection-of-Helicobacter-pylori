import torch

def rgb_to_hsv_torch(img):
    """
    img: tensor (B, 3, H, W) con valores en [0,1]
    return: hsv tensor (B, 3, H, W)
    """

    r, g, b = img[:,0], img[:,1], img[:,2]

    maxc = torch.max(img, dim=1).values
    minc = torch.min(img, dim=1).values
    v = maxc

    deltac = maxc - minc
    s = deltac / (maxc + 1e-8)

    # Hue calculation
    rc = (maxc - r) / (deltac + 1e-8)
    gc = (maxc - g) / (deltac + 1e-8)
    bc = (maxc - b) / (deltac + 1e-8)

    h = torch.zeros_like(maxc)

    mask = (maxc == r)
    h[mask] = (bc - gc)[mask]

    mask = (maxc == g)
    h[mask] = 2.0 + (rc - bc)[mask]

    mask = (maxc == b)
    h[mask] = 4.0 + (gc - rc)[mask]

    h = (h / 6.0) % 1.0

    hsv = torch.stack([h, s, v], dim=1)
    return hsv