"""
Grad-CAM implementation as a context manager.

Usage:
    with GradCAM(model, model.layer4) as cam:
        heatmap, class_idx, prob = cam(input_tensor)

The context manager form guarantees the forward/backward hooks are removed
even if an exception is raised — important in a long-running app process.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle = None
        self._bwd_handle = None

    # ----- context-manager plumbing -----

    def __enter__(self) -> "GradCAM":
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()

    def remove(self) -> None:
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None

    # ----- hooks -----

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] is dL/dA — the quantity Grad-CAM needs.
        self.gradients = grad_output[0].detach()

    # ----- the actual computation -----

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Run Grad-CAM. Returns (heatmap HxW in [0,1], class_idx, class probability).

        `input_tensor` shape: (1, 3, H, W), already normalized, on the same
        device as the model.
        """
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError(
                f"Expected (1,3,H,W) tensor, got shape {tuple(input_tensor.shape)}"
            )

        self.model.eval()
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward pass through the chosen class logit triggers the gradient hook.
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Hooks did not fire. Did you wrap the wrong layer or forget the "
                "`with` block?"
            )

        # Average gradients across spatial dims → one importance weight per channel.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)            # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)        # (1, 1, h, w)
        cam = torch.relu(cam)

        # Upsample to input resolution and min-max normalize to [0, 1].
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False,
        )
        cam_np = cam.squeeze().cpu().numpy()
        if cam_np.max() > 0:
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

        prob = float(probs[0, class_idx].item())
        return cam_np, int(class_idx), prob
