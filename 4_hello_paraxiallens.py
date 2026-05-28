"""Hello, world! for DeepLens ParaxialLens class.

In this code, we construct a paraxial (thin-lens / ABCD-matrix) lens. This
simple model simulates defocus blur via the circle of confusion (CoC) but not
higher-order optical aberrations. It is a fast baseline renderer for
depth-of-field effects, as commonly used in Blender and similar tools.

We refocus the lens, inspect the circle of confusion and depth of field at a few
depths, generate a defocus PSF, and render a synthetic RGB-D image with
out-of-focus blur.

Reference:
    [1] https://en.wikipedia.org/wiki/Circle_of_confusion
    [2] https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
"""

import torch
from torchvision.utils import save_image

from deeplens import ParaxialLens

# =====================================================================
# Lens construction and focusing
# =====================================================================
# A 50 mm f/1.8 lens on a 20 x 20 mm sensor.
lens = ParaxialLens(
    foclen=50.0,
    fnum=1.8,
    sensor_size=(20.0, 20.0),
    sensor_res=(64, 64),
    device="cpu",
)

# Focus the lens at 1 m in front of the camera (depths are negative, in mm).
lens.refocus(-1000.0)
print(f"ParaxialLens: f={lens.foclen} mm, f/{lens.fnum}, focused at {lens.foc_dist} mm.")

# =====================================================================
# Defocus analysis: circle of confusion (CoC) and depth of field (DoF)
# =====================================================================
depths = torch.tensor([-500.0, -1000.0, -2000.0])  # near / in-focus / far
coc = lens.coc(depths)
dof = lens.dof(depths)
for d, c, f in zip(depths.tolist(), coc.tolist(), dof.tolist()):
    print(f"  depth {d:8.1f} mm -> CoC {c:7.4f} mm, DoF {f:8.2f} mm")
# CoC is ~0 at the focus distance and grows for out-of-focus depths.

# =====================================================================
# PSF and image simulation
# =====================================================================
save_name = "./hello_paraxiallens"

# A defocused on-axis point source produces a blur disk (pillbox) PSF.
point = torch.tensor([[0.0, 0.0, -500.0]])
psf = lens.psf(point, ks=31, psf_type="pillbox")
print(f"Defocus PSF: shape {tuple(psf.shape[-2:])}, sum {psf.sum():.3f}")
save_image(psf.clamp(min=0), f"{save_name}_psf.png", normalize=True)

# Render a synthetic scene: a random RGB image at a uniform out-of-focus depth.
rgb = torch.rand(1, 3, 64, 64)
depth_map = torch.full((1, 1, 64, 64), 500.0)  # object-space depth [mm]
img = lens.render_rgbd(rgb, depth_map)
print(f"Rendered RGB-D image: shape {tuple(img.shape)}")
save_image(img.clamp(0, 1), f"{save_name}_render.png")
print(f"Saved outputs to {save_name}_psf.png and {save_name}_render.png")
