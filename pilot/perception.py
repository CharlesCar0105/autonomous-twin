"""
perception.py -- Segmentation route (seuillage ou U-Net).

Le U-Net entraine sur le dataset blanc/noir a appris exactement un
seuillage de luminance (diff=0 vs GT sur le set val). Par simplicite
on commence par le seuillage direct ici. Le vrai U-Net peut etre
branche via la fonction `load_unet` plus tard si le style visuel
devient plus varie.
"""

from typing import Optional
import numpy as np

_unet_model = None
_unet_device = None


def compute_mask(camera: np.ndarray, size: Optional[int] = None) -> np.ndarray:
    """Calcule le masque route binaire a partir d'une image camera RGB.

    Utilise un seuillage simple (pixel blanc = route) par defaut. Resize
    optionnel a `size x size` (bilinear) si fourni.

    Args:
        camera : (H, W, 3) uint8 ou float
        size   : taille cible carree, None = pas de resize

    Returns:
        mask   : (size, size) ou (H, W) float32 dans {0.0, 1.0}
    """
    cam = np.asarray(camera)
    if cam.dtype != np.uint8:
        cam = (cam * 255).astype(np.uint8) if cam.max() <= 1.5 else cam.astype(np.uint8)
    r, g, b = cam[..., 0], cam[..., 1], cam[..., 2]
    mask = ((r > 200) & (g > 200) & (b > 200)).astype(np.float32)
    if size is not None and mask.shape[0] != size:
        mask = _resize_nearest(mask, size, size)
    return mask


def _resize_nearest(a: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize sans dependance externe."""
    h, w = a.shape
    ys = (np.arange(new_h) * h / new_h).astype(np.int64)
    xs = (np.arange(new_w) * w / new_w).astype(np.int64)
    return a[ys[:, None], xs[None, :]]


def load_unet(weights_path: str) -> None:
    """Charge un U-Net entraine a la place du seuillage (facultatif,
    utile si le dataset devient plus varie visuellement)."""
    global _unet_model, _unet_device
    import torch
    _unet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Import local pour ne pas exiger torch quand on utilise seuillage.
    from pilot.unet_arch import UNet
    _unet_model = UNet(base=32).to(_unet_device)
    _unet_model.load_state_dict(torch.load(weights_path, map_location=_unet_device))
    _unet_model.eval()


def compute_mask_unet(camera: np.ndarray, size: Optional[int] = None) -> np.ndarray:
    """Inference U-Net. Necessite un load_unet() prealable."""
    import torch
    if _unet_model is None:
        raise RuntimeError("load_unet(...) requis avant compute_mask_unet")
    cam = np.asarray(camera, dtype=np.float32)
    if cam.max() > 1.5:
        cam = cam / 255.0
    t = torch.from_numpy(cam).permute(2, 0, 1).unsqueeze(0).to(_unet_device)
    with torch.no_grad():
        probs = torch.sigmoid(_unet_model(t)).cpu().squeeze().numpy()
    mask = (probs > 0.5).astype(np.float32)
    if size is not None and mask.shape[0] != size:
        mask = _resize_nearest(mask, size, size)
    return mask
