"""Hello, world! for DeepLens HybridLens class.

A hybrid lens combines a refractive GeoLens with a diffractive optical element
(DOE) placed behind it. A differentiable ray-wave model is used: coherent ray
tracing computes the complex wavefront at the DOE plane (capturing geometric
aberrations), the DOE modulates the phase, and the Angular Spectrum Method
propagates the field to the sensor.

This is a MINIMAL intro: we load a hybrid lens, draw its layout, and compute a
single on-axis PSF. For the full end-to-end joint optimization loop, see
6_hybridlens_design.py.

Note:
    HybridLens runs in float64 for accurate phase tracing.

Technical Paper:
    Xinge Yang, Matheus Souza, Kunyi Wang, Praneeth Chakravarthula, Qiang Fu,
    Wolfgang Heidrich, "End-to-End Hybrid Refractive-Diffractive Lens Design
    with Differentiable Ray-Wave Model," SIGGRAPH Asia 2024.
"""

import torch

from deeplens import HybridLens

# HybridLens requires float64 as the default dtype for accurate phase tracing.
torch.set_default_dtype(torch.float64)

# =====================================================================
# Lens loading
# =====================================================================
# Load an example hybrid lens (an A489 refractive design + a Binary2 DOE).
lens = HybridLens(filename="./datasets/lenses/hybridlens/a489_doe.json")
print(f"HybridLens: {len(lens.geolens.surfaces)} refractive surface(s) + "
      f"a {type(lens.doe).__name__} DOE.")

# Focus the lens at 1 m (depths are negative, in mm).
lens.refocus(foc_dist=-1000.0)

# =====================================================================
# Layout and PSF analysis
# =====================================================================
save_name = "./hello_hybridlens"

# Draw the lens layout: refractive elements, traced rays, and the
# DOE-to-sensor wave-propagation region.
lens.draw_layout(save_name=f"{save_name}_layout.png")
print(f"Saved lens layout to {save_name}_layout.png")

# Compute a single on-axis PSF. The ray-wave model captures the contribution of
# all diffraction orders at once. Coherent ray tracing needs >= 1e6 samples.
psf = lens.psf(points=[0.0, 0.0, -10000.0], ks=64, spp=1_000_000)
print(f"On-axis PSF: shape {tuple(psf.shape)}, sum {psf.sum():.3f}")
