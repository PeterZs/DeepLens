"""Hello, world! for DeepLens DiffractiveLens class.

In this code, we construct a paraxial diffractive lens (a single Fresnel DOE in
front of the sensor) from scratch. Each optical element is modelled as a phase
function and the wavefront is propagated to the sensor with the Angular Spectrum
Method (ASM). We then compute on-axis PSFs for an object at infinity and at a
finite depth, and save them as images.

Note:
    DiffractiveLens runs in float64 for numerical stability of the wave
    propagation step, and the PSF is computed on-axis (paraxial approximation).

Technical Paper:
    [1] Vincent Sitzmann et al., "End-to-end optimization of optics and image
        processing for achromatic extended depth of field and super-resolution
        imaging," SIGGRAPH 2018.
    [2] Qilin Sun et al., "Learning Rank-1 Diffractive Optics for Single-shot
        High Dynamic Range Imaging," CVPR 2020.
"""

import torch
from torchvision.utils import save_image

from deeplens import DiffractiveLens
from deeplens.diffractive_surface import Fresnel

# =====================================================================
# Lens construction
# =====================================================================
# Build a minimal diffractive lens programmatically: a single Fresnel DOE
# focusing at f0 = 50 mm, placed one focal length in front of the sensor.
# (We construct it by hand because the wave model is most transparent this way.)
lens = DiffractiveLens(device="cpu")
lens.surfaces = [Fresnel(f0=50, d=0, res=500, fab_ps=0.008)]
lens.d_sensor = torch.tensor(50.0, dtype=torch.float64)
lens.sensor_size = (4.0, 4.0)
lens.sensor_res = (500, 500)
lens.pixel_size = lens.sensor_size[0] / lens.sensor_res[0]
lens.surfaces[0].to(lens.device)

print(f"DiffractiveLens with {len(lens.surfaces)} surface(s), "
      f"sensor {lens.sensor_size} mm @ {lens.sensor_res} px.")

# =====================================================================
# PSF analysis
# =====================================================================
save_name = "./hello_diffraclens"
ks = 128

# On-axis PSF for an object at infinity (plane wave input).
psf_inf = lens.psf(depth=float("inf"), ks=ks)
print(f"Infinity-focus PSF: shape {tuple(psf_inf.shape)}, sum {psf_inf.sum():.3f}")

# On-axis PSF for a finite object depth (point-source / spherical wave input).
psf_near = lens.psf(depth=-500.0, ks=ks)
print(f"Finite-depth PSF:  shape {tuple(psf_near.shape)}, sum {psf_near.sum():.3f}")

# Save the PSFs as images (normalized for visualization).
save_image(psf_inf[None].clamp(min=0), f"{save_name}_psf_inf.png", normalize=True)
save_image(psf_near[None].clamp(min=0), f"{save_name}_psf_near.png", normalize=True)
print(f"Saved PSF images to {save_name}_psf_inf.png and {save_name}_psf_near.png")
