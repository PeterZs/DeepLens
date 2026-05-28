# Three Paper-Based Diffractive Surfaces — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three paper-based DOE parameterizations (`Rank1`, `DiffractedRotation`, `RotationallySymmetric`) to `deeplens/diffractive_surface/`, integrate them into `DiffractiveLens`, and verify their signature PSF behavior on GPU.

**Architecture:** Each class subclasses `DiffractiveSurface` and implements `phase_func()` (raw phase at `wvln0`); the base class handles wrapping/quantization, per-wavelength scaling, propagation, and IO. Register in the package `__init__.py` and the `DiffractiveLens.read_lens_json` dispatcher. No reconstruction networks; `HybridLens` is untouched.

**Tech Stack:** Python, PyTorch, pytest. Wave-optics PSF via the existing `DiffractiveLens` ASM model.

**Spec:** `docs/superpowers/specs/2026-05-28-diffractive-surfaces-design.md`

**Conventions (read before starting):**
- `DiffractiveSurface.__init__(d, res, fab_ps=0.001, fab_step=16, wvln0=0.55, mat="fused_silica", design_ps=None, is_square=True, device="cpu")`. `self.res=(res,res)` if int. `self.x, self.y` are `[res[0], res[1]]` grids (mm). `self.w=res[0]*ps`, `self.h=res[1]*ps`.
- `phase_func()` returns an `[res[0], res[1]]` tensor of **raw phase at `wvln0`** (may exceed 2π; the base wraps + quantizes).
- Params are plain tensors (NOT `nn.Parameter`); `get_optimizer_params` sets `requires_grad=True` and returns `[{"params": [...], "lr": ...}]`.
- New files need the Apache copyright header (copy from `fresnel.py` lines 1-5).

---

### Task 1: `Rank1` surface

**Files:**
- Create: `deeplens/diffractive_surface/rank1.py`
- Modify: `deeplens/diffractive_surface/__init__.py`
- Modify: `deeplens/diffraclens.py` (import + loader branch + `write_lens_json` path handling)
- Create: `datasets/lenses/diffraclens/rank1.json`
- Test: `test/test_diffractive_surfaces.py`

- [ ] **Step 1: Write failing tests** — append to `test/test_diffractive_surfaces.py`:

```python
from deeplens.diffractive_surface import Rank1  # add to the existing import block


class TestRank1:
    """Tests for Rank1 DOE."""

    def test_init(self):
        doe = Rank1(d=0.0, rank=1, res=100)
        assert doe.res == (100, 100)
        assert doe.V.shape == (100, 1)
        assert doe.Q.shape == (100, 1)

    def test_phase_func_shape(self):
        doe = Rank1(d=0.0, rank=1, res=100)
        phase = doe.phase_func()
        assert phase.shape == (100, 100)

    def test_height_is_low_rank(self):
        """The pre-sigmoid height logits are exactly rank == `rank`."""
        doe = Rank1(d=0.0, rank=1, res=100)
        assert torch.linalg.matrix_rank(doe.V @ doe.Q.T) == 1
        doe3 = Rank1(d=0.0, rank=3, res=100)
        assert doe3.V.shape == (100, 3)
        assert torch.linalg.matrix_rank(doe3.V @ doe3.Q.T) == 3

    def test_optimizer_params(self):
        doe = Rank1(d=0.0, rank=1, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 1
        assert doe.V.requires_grad
        assert doe.Q.requires_grad
```

- [ ] **Step 2: Run, verify it fails** — `pytest test/test_diffractive_surfaces.py::TestRank1 -q`. Expected: ImportError (`Rank1` not exported).

- [ ] **Step 3: Implement** `deeplens/diffractive_surface/rank1.py`:

```python
# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Rank-1 (low-rank) DOE parameterization.

The height map is a low-rank outer product ``h = h_max * sigmoid(V @ Q.T)``
(default rank 1). Because ``h_max`` corresponds to a 2*pi phase shift at the
design wavelength, the design-wavelength phase is ``2*pi * sigmoid(V @ Q.T)``.

Reference:
    Qilin Sun, Ethan Tseng, Qiang Fu, Wolfgang Heidrich, Felix Heide,
    "Learning Rank-1 Diffractive Optics for Single-shot High Dynamic Range
    Imaging," CVPR 2020.
"""

import torch

from .diffractive import DiffractiveSurface


class Rank1(DiffractiveSurface):
    """DOE whose height map is constrained to a low-rank outer product."""

    def __init__(
        self,
        d,
        rank=1,
        V=None,
        Q=None,
        res=(1000, 1000),
        mat="fused_silica",
        wvln0=0.55,
        fab_ps=0.001,
        fab_step=16,
        is_square=True,
        device="cpu",
    ):
        """Initialize a rank-`rank` DOE.

        Args:
            d (float): Distance of the DOE surface. [mm]
            rank (int): Rank of the height map (default 1).
            V (Tensor, optional): Left factor, shape [res[0], rank].
            Q (Tensor, optional): Right factor, shape [res[1], rank].
            res (tuple or int): DOE resolution [w, h]. [pixel]
            mat (str): DOE material.
            wvln0 (float): Design wavelength. [um]
            fab_ps (float): Fabrication pixel size. [mm]
            fab_step (int): Quantization levels.
            device (str): Compute device.
        """
        super().__init__(
            d=d, res=res, mat=mat, wvln0=wvln0, fab_ps=fab_ps,
            fab_step=fab_step, is_square=is_square, device=device,
        )
        self.rank = rank
        self.V = torch.randn(self.res[0], rank) * 1e-3 if V is None else V
        self.Q = torch.randn(self.res[1], rank) * 1e-3 if Q is None else Q
        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize Rank1 DOE from a dict."""
        V = Q = None
        weight_path = doe_dict.get("weight_path", None)
        if weight_path is not None:
            w = torch.load(weight_path, weights_only=True)
            V, Q = w["V"], w["Q"]
        return cls(
            d=doe_dict["d"],
            rank=doe_dict.get("rank", 1),
            V=V,
            Q=Q,
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            wvln0=doe_dict.get("wvln0", 0.55),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            is_square=doe_dict.get("is_square", True),
        )

    def phase_func(self):
        """Get the raw phase map at the design wavelength."""
        return 2 * torch.pi * torch.sigmoid(self.V @ self.Q.T)

    # ===== Optimization =====
    def get_optimizer_params(self, lr=0.01):
        """Get parameters for optimization."""
        self.V.requires_grad = True
        self.Q.requires_grad = True
        return [{"params": [self.V, self.Q], "lr": lr}]

    # ===== IO =====
    def surf_dict(self, weight_path):
        """Return a dict of surface; saves [V, Q] to `weight_path`."""
        surf_dict = super().surf_dict()
        surf_dict["rank"] = self.rank
        surf_dict["weight_path"] = weight_path
        torch.save(
            {"V": self.V.clone().detach().cpu(), "Q": self.Q.clone().detach().cpu()},
            weight_path,
        )
        return surf_dict
```

- [ ] **Step 4: Register** in `deeplens/diffractive_surface/__init__.py` — add `from .rank1 import Rank1` after the `pixel2d` import, and add `"Rank1"` to `__all__`.

- [ ] **Step 5: Run tests, verify pass** — `pytest test/test_diffractive_surfaces.py::TestRank1 -q`. Expected: 4 passed.

- [ ] **Step 6: Wire into the loader** — in `deeplens/diffraclens.py`:
  1. Extend the import block (lines ~24-30) to include `Rank1` (and, added in later tasks, `DiffractedRotation`, `RotationallySymmetric`).
  2. In `read_lens_json`, after the `zernike` branch (line ~165) add:

```python
                elif surf_dict["type"].lower() == "rank1":
                    s = Rank1.init_from_dict(surf_dict)
```

  3. In `write_lens_json`, change the Pixel2D special-case (line ~198) to also handle path-based surfaces:

```python
            if isinstance(s, (Pixel2D, Rank1)):
                surf_data = s.surf_dict(filename.replace(".json", f"_surf{i + 1}.pth"))
            else:
                surf_data = s.surf_dict()
```

  (Task 3 extends this tuple with `RotationallySymmetric`.)

- [ ] **Step 7: Create** `datasets/lenses/diffraclens/rank1.json` (Fresnel focusing lens + Rank1 encoder, so the loaded lens focuses):

```json
{
    "info": "Fresnel focusing lens + Rank-1 encoding DOE (Sun et al., CVPR 2020) demo.",
    "d_sensor": 50.0,
    "sensor_size": [4.0, 4.0],
    "sensor_res": [500, 500],
    "surfaces": [
        {"idx": 1, "type": "Fresnel", "f0": 50.0, "res": [500, 500], "fab_ps": 0.008, "wvln0": 0.55, "d_next": 0.0},
        {"idx": 2, "type": "Rank1", "rank": 1, "res": [500, 500], "fab_ps": 0.008, "wvln0": 0.55, "d_next": 50.0}
    ]
}
```

- [ ] **Step 8: Add a load test** — append to `test/test_diffractive_surfaces.py`:

```python
class TestDiffractiveLensLoad:
    """The new surfaces load from JSON via DiffractiveLens and produce a PSF."""

    def test_load_rank1(self):
        from deeplens import DiffractiveLens

        lens = DiffractiveLens(filename="./datasets/lenses/diffraclens/rank1.json")
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=32)
        assert psf.shape == (32, 32)
        assert torch.isfinite(psf).all()
```

- [ ] **Step 9: Run, verify pass** — `pytest test/test_diffractive_surfaces.py::TestRank1 test/test_diffractive_surfaces.py::TestDiffractiveLensLoad::test_load_rank1 -q`. Expected: all passed.

- [ ] **Step 10: Commit**

```bash
git add deeplens/diffractive_surface/rank1.py deeplens/diffractive_surface/__init__.py deeplens/diffraclens.py datasets/lenses/diffraclens/rank1.json test/test_diffractive_surfaces.py
git commit -m "feat(diffractive): add Rank1 low-rank DOE surface (Sun et al. CVPR 2020)"
```

---

### Task 2: `DiffractedRotation` surface

**Files:**
- Create: `deeplens/diffractive_surface/diffracted_rotation.py`
- Modify: `deeplens/diffractive_surface/__init__.py`
- Modify: `deeplens/diffraclens.py` (import + loader branch)
- Create: `datasets/lenses/diffraclens/diffracted_rotation.json`
- Test: `test/test_diffractive_surfaces.py`

- [ ] **Step 1: Write failing tests** — append:

```python
from deeplens.diffractive_surface import DiffractedRotation  # add to import block


class TestDiffractedRotation:
    """Tests for DiffractedRotation DOE."""

    def test_init(self):
        doe = DiffractedRotation(d=0.0, f0=50.0, num_wings=3, res=100)
        assert doe.res == (100, 100)
        assert doe.num_wings == 3
        assert doe.wvln0 == pytest.approx(0.66)  # defaults to wvln_max

    def test_phase_func_shape(self):
        doe = DiffractedRotation(d=0.0, f0=50.0, res=100)
        assert doe.phase_func().shape == (100, 100)

    def test_phase_is_anisotropic(self):
        """The rotating DOE is NOT transpose-symmetric (unlike a radial lens)."""
        doe = DiffractedRotation(d=0.0, f0=50.0, num_wings=3, res=128)
        phase = doe.phase_func()
        assert not torch.allclose(phase, phase.T, atol=1e-3)

    def test_optimizer_params(self):
        doe = DiffractedRotation(d=0.0, f0=50.0, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 1
        assert doe.f0.requires_grad
```

- [ ] **Step 2: Run, verify fail** — `pytest test/test_diffractive_surfaces.py::TestDiffractedRotation -q`. Expected: ImportError.

- [ ] **Step 3: Implement** `deeplens/diffractive_surface/diffracted_rotation.py`:

```python
# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Diffracted-rotation DOE for snapshot hyperspectral imaging.

Each angular wedge is a Fresnel lens blazed for a different "matched"
wavelength, so the focused PSF is an anisotropic lobe whose orientation
rotates monotonically with wavelength.

Reference:
    Daniel S. Jeon, Seung-Hwan Baek, Shinyoung Yi, Qiang Fu, Xiong Dun,
    Wolfgang Heidrich, Min H. Kim, "Compact Snapshot Hyperspectral Imaging
    with Diffracted Rotation," ACM TOG (SIGGRAPH) 2019.
"""

import torch

from .diffractive import DiffractiveSurface


class DiffractedRotation(DiffractiveSurface):
    """Analytic spiral DOE: per-wedge Fresnel lens with angular wavelength match."""

    def __init__(
        self,
        d,
        f0,
        num_wings=3,
        wvln_min=0.42,
        wvln_max=0.66,
        wvln0=None,
        res=(1000, 1000),
        mat="fused_silica",
        fab_ps=0.001,
        fab_step=16,
        is_square=True,
        circular=True,
        device="cpu",
    ):
        """Initialize a diffracted-rotation DOE.

        Args:
            d (float): Distance of the DOE surface. [mm]
            f0 (float): Focal length. [mm]
            num_wings (int): Number of angular wings N (fixed design choice).
            wvln_min (float): Min matched wavelength. [um]
            wvln_max (float): Max matched wavelength. [um]
            wvln0 (float, optional): Design wavelength; defaults to ``wvln_max``
                so the wrapped phase never exceeds 2*pi.
            res (tuple or int): DOE resolution. [pixel]
            mat (str): DOE material.
            fab_ps (float): Fabrication pixel size. [mm]
            fab_step (int): Quantization levels.
            circular (bool): Zero the phase outside the inscribed circle.
            device (str): Compute device.
        """
        if wvln0 is None:
            wvln0 = wvln_max
        super().__init__(
            d=d, res=res, mat=mat, wvln0=wvln0, fab_ps=fab_ps,
            fab_step=fab_step, is_square=is_square, device=device,
        )
        self.f0 = f0 if torch.is_tensor(f0) else torch.tensor(float(f0))
        self.num_wings = num_wings
        self.wvln_min = wvln_min
        self.wvln_max = wvln_max
        self.circular = circular

        # Cache static polar grids.
        self.r2 = self.x**2 + self.y**2
        self.theta = torch.remainder(torch.atan2(self.y, self.x), 2 * torch.pi)
        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize DiffractedRotation DOE from a dict."""
        return cls(
            d=doe_dict["d"],
            f0=doe_dict["f0"],
            num_wings=doe_dict.get("num_wings", 3),
            wvln_min=doe_dict.get("wvln_min", 0.42),
            wvln_max=doe_dict.get("wvln_max", 0.66),
            wvln0=doe_dict.get("wvln0", None),
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            circular=doe_dict.get("circular", True),
        )

    def phase_func(self):
        """Get the raw phase map at the design wavelength."""
        # Ideal converging-lens optical path difference [mm].
        opd = torch.sqrt(self.r2 + self.f0**2) - self.f0
        # Matched wavelength per angle (sawtooth, num_wings periods over 2*pi) [mm].
        frac = torch.remainder(self.theta * self.num_wings / (2 * torch.pi), 1.0)
        lam_m_mm = (self.wvln_min + (self.wvln_max - self.wvln_min) * frac) * 1e-3
        wvln0_mm = self.wvln0 * 1e-3
        # Blaze each wedge for its matched wavelength.
        phase = (2 * torch.pi / wvln0_mm) * torch.remainder(opd, lam_m_mm)
        if self.circular:
            r_max = min(self.w, self.h) / 2
            phase = torch.where(
                self.r2 <= r_max**2, phase, torch.zeros_like(phase)
            )
        return phase

    # ===== Optimization =====
    def get_optimizer_params(self, lr=0.001):
        """Get parameters for optimization (focal length)."""
        self.f0.requires_grad = True
        return [{"params": [self.f0], "lr": lr}]

    # ===== IO =====
    def surf_dict(self):
        """Return a dict of surface."""
        surf_dict = super().surf_dict()
        surf_dict["f0"] = round(self.f0.item(), 4)
        surf_dict["num_wings"] = self.num_wings
        surf_dict["wvln_min"] = self.wvln_min
        surf_dict["wvln_max"] = self.wvln_max
        return surf_dict
```

- [ ] **Step 4: Register** in `__init__.py` — add `from .diffracted_rotation import DiffractedRotation` and `"DiffractedRotation"` to `__all__`.

- [ ] **Step 5: Loader** — in `deeplens/diffraclens.py` add `DiffractedRotation` to the import block and add after the `rank1` branch:

```python
                elif surf_dict["type"].lower() == "diffractedrotation":
                    s = DiffractedRotation.init_from_dict(surf_dict)
```

- [ ] **Step 6: Create** `datasets/lenses/diffracted_rotation` config `datasets/lenses/diffraclens/diffracted_rotation.json`:

```json
{
    "info": "Diffracted-rotation hyperspectral DOE (Jeon et al., TOG 2019) demo.",
    "d_sensor": 50.0,
    "sensor_size": [1.0, 1.0],
    "sensor_res": [500, 500],
    "surfaces": [
        {"idx": 1, "type": "DiffractedRotation", "f0": 50.0, "num_wings": 3, "wvln_min": 0.42, "wvln_max": 0.66, "res": [1000, 1000], "fab_ps": 0.001, "d_next": 50.0}
    ]
}
```

- [ ] **Step 7: Add a load test** — append to `TestDiffractiveLensLoad`:

```python
    def test_load_diffracted_rotation(self):
        from deeplens import DiffractiveLens

        lens = DiffractiveLens(filename="./datasets/lenses/diffraclens/diffracted_rotation.json")
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=32, wvln=0.55)
        assert psf.shape == (32, 32)
        assert torch.isfinite(psf).all()
```

- [ ] **Step 8: Run, verify pass** — `pytest test/test_diffractive_surfaces.py::TestDiffractedRotation test/test_diffractive_surfaces.py::TestDiffractiveLensLoad::test_load_diffracted_rotation -q`. Expected: all passed.

- [ ] **Step 9: Commit**

```bash
git add deeplens/diffractive_surface/diffracted_rotation.py deeplens/diffractive_surface/__init__.py deeplens/diffraclens.py datasets/lenses/diffraclens/diffracted_rotation.json test/test_diffractive_surfaces.py
git commit -m "feat(diffractive): add DiffractedRotation hyperspectral DOE (Jeon et al. TOG 2019)"
```

---

### Task 3: `RotationallySymmetric` surface

**Files:**
- Create: `deeplens/diffractive_surface/rotational_symmetric.py`
- Modify: `deeplens/diffractive_surface/__init__.py`
- Modify: `deeplens/diffraclens.py` (import + loader branch + `write_lens_json` tuple)
- Create: `datasets/lenses/diffraclens/rotational_symmetric.json`
- Test: `test/test_diffractive_surfaces.py`

- [ ] **Step 1: Write failing tests** — append:

```python
from deeplens.diffractive_surface import RotationallySymmetric  # add to import block


class TestRotationallySymmetric:
    """Tests for RotationallySymmetric DOE."""

    def test_init(self):
        doe = RotationallySymmetric(d=0.0, f0=50.0, res=100)
        assert doe.res == (100, 100)
        assert doe.n_rings == 50
        assert doe.radial_phase.shape == (50,)

    def test_phase_func_shape(self):
        doe = RotationallySymmetric(d=0.0, f0=50.0, res=100)
        assert doe.phase_func().shape == (100, 100)

    def test_phase_is_radially_symmetric(self):
        """Phase depends only on radius => transpose-symmetric on a square grid."""
        doe = RotationallySymmetric(d=0.0, f0=50.0, res=128)
        phase = doe.phase_func()
        assert torch.allclose(phase, phase.T, atol=1e-4)

    def test_optimizer_params(self):
        doe = RotationallySymmetric(d=0.0, f0=50.0, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 1
        assert doe.radial_phase.requires_grad
```

- [ ] **Step 2: Run, verify fail** — `pytest test/test_diffractive_surfaces.py::TestRotationallySymmetric -q`. Expected: ImportError.

- [ ] **Step 3: Implement** `deeplens/diffractive_surface/rotational_symmetric.py`:

```python
# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Rotationally symmetric DOE parameterized by a free-form 1D radial profile.

The phase is defined by a 1D radial vector ``radial_phase`` of ``n_rings``
samples and broadcast to 2D by ``r = sqrt(x**2 + y**2)`` via differentiable
linear interpolation across rings.

Reference:
    Xiong Dun, Hayato Ikoma, Gordon Wetzstein, Zhanshan Wang, Xinbin Cheng,
    Yifan Peng, "Learned rotationally symmetric diffractive achromat for
    full-spectrum computational imaging," Optica 2020.
"""

import torch

from .diffractive import DiffractiveSurface


class RotationallySymmetric(DiffractiveSurface):
    """DOE defined by a 1D radial phase profile (rotationally symmetric)."""

    def __init__(
        self,
        d,
        f0=None,
        n_rings=None,
        init="fresnel",
        radial_phase=None,
        res=(1000, 1000),
        mat="fused_silica",
        wvln0=0.55,
        fab_ps=0.001,
        fab_step=16,
        is_square=True,
        circular=True,
        device="cpu",
    ):
        """Initialize a rotationally symmetric DOE.

        Args:
            d (float): Distance of the DOE surface. [mm]
            f0 (float, optional): Focal length for ``init="fresnel"``. [mm]
            n_rings (int, optional): Number of radial samples; defaults to res[0]//2.
            init (str): "fresnel" (Fresnel radial profile) or "flat".
            radial_phase (Tensor, optional): Explicit 1D radial phase [n_rings].
            res (tuple or int): DOE resolution. [pixel]
            mat (str): DOE material.
            wvln0 (float): Design wavelength. [um]
            fab_ps (float): Fabrication pixel size. [mm]
            fab_step (int): Quantization levels.
            circular (bool): Zero the phase outside the inscribed circle.
            device (str): Compute device.
        """
        super().__init__(
            d=d, res=res, mat=mat, wvln0=wvln0, fab_ps=fab_ps,
            fab_step=fab_step, is_square=is_square, device=device,
        )
        self.n_rings = self.res[0] // 2 if n_rings is None else n_rings
        self.circular = circular
        self.r_max = min(self.w, self.h) / 2  # inscribed radius [mm]

        # Cache radial-interpolation indices/weights (function of r only).
        r = torch.sqrt(self.x**2 + self.y**2)
        t = (r / self.r_max).clamp(0, 1) * (self.n_rings - 1)
        self.idx0 = torch.floor(t).long().clamp(0, self.n_rings - 1)
        self.idx1 = (self.idx0 + 1).clamp(0, self.n_rings - 1)
        self.frac = t - self.idx0.to(t.dtype)
        self.r_grid = r

        # Initialize the 1D radial phase profile.
        if radial_phase is not None:
            self.radial_phase = radial_phase
        elif init == "fresnel":
            assert f0 is not None, "init='fresnel' requires f0."
            ring_r = torch.linspace(0, self.r_max, self.n_rings)
            wvln0_mm = wvln0 * 1e-3
            self.radial_phase = -torch.pi * ring_r**2 / (float(f0) * wvln0_mm)
        elif init == "flat":
            self.radial_phase = torch.ones(self.n_rings) * 1e-3
        else:
            raise ValueError(f"Unknown init: {init}")

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize RotationallySymmetric DOE from a dict."""
        radial_phase = None
        weight_path = doe_dict.get("weight_path", None)
        if weight_path is not None:
            radial_phase = torch.load(weight_path, weights_only=True)
        return cls(
            d=doe_dict["d"],
            f0=doe_dict.get("f0", None),
            n_rings=doe_dict.get("n_rings", None),
            init=doe_dict.get("init", "fresnel"),
            radial_phase=radial_phase,
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            wvln0=doe_dict.get("wvln0", 0.55),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            circular=doe_dict.get("circular", True),
        )

    def phase_func(self):
        """Get the raw phase map at the design wavelength."""
        # Differentiable linear interpolation of the 1D profile onto the 2D grid.
        phase = (
            self.radial_phase[self.idx0] * (1 - self.frac)
            + self.radial_phase[self.idx1] * self.frac
        )
        if self.circular:
            phase = torch.where(
                self.r_grid <= self.r_max, phase, torch.zeros_like(phase)
            )
        return phase

    # ===== Optimization =====
    def get_optimizer_params(self, lr=0.01):
        """Get parameters for optimization (radial profile)."""
        self.radial_phase.requires_grad = True
        return [{"params": [self.radial_phase], "lr": lr}]

    # ===== IO =====
    def surf_dict(self, weight_path):
        """Return a dict of surface; saves the radial profile to `weight_path`."""
        surf_dict = super().surf_dict()
        surf_dict["n_rings"] = self.n_rings
        surf_dict["weight_path"] = weight_path
        torch.save(self.radial_phase.clone().detach().cpu(), weight_path)
        return surf_dict
```

- [ ] **Step 4: Register** in `__init__.py` — add `from .rotational_symmetric import RotationallySymmetric` and `"RotationallySymmetric"` to `__all__`.

- [ ] **Step 5: Loader** — in `deeplens/diffraclens.py` add `RotationallySymmetric` to the import block; add after the `diffractedrotation` branch:

```python
                elif surf_dict["type"].lower() == "rotationallysymmetric":
                    s = RotationallySymmetric.init_from_dict(surf_dict)
```

  and extend the `write_lens_json` tuple to `isinstance(s, (Pixel2D, Rank1, RotationallySymmetric))`.

- [ ] **Step 6: Create** `datasets/lenses/diffraclens/rotational_symmetric.json`:

```json
{
    "info": "Rotationally symmetric diffractive achromat (Dun et al., Optica 2020) demo.",
    "d_sensor": 50.0,
    "sensor_size": [4.0, 4.0],
    "sensor_res": [500, 500],
    "surfaces": [
        {"idx": 1, "type": "RotationallySymmetric", "f0": 50.0, "init": "fresnel", "res": [1000, 1000], "fab_ps": 0.004, "wvln0": 0.55, "d_next": 50.0}
    ]
}
```

- [ ] **Step 7: Add a load test** — append to `TestDiffractiveLensLoad`:

```python
    def test_load_rotational_symmetric(self):
        from deeplens import DiffractiveLens

        lens = DiffractiveLens(filename="./datasets/lenses/diffraclens/rotational_symmetric.json")
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=32)
        assert psf.shape == (32, 32)
        assert torch.isfinite(psf).all()
```

- [ ] **Step 8: Run, verify pass** — `pytest test/test_diffractive_surfaces.py -q`. Expected: all tests (old + new) pass.

- [ ] **Step 9: Commit**

```bash
git add deeplens/diffractive_surface/rotational_symmetric.py deeplens/diffractive_surface/__init__.py deeplens/diffraclens.py datasets/lenses/diffraclens/rotational_symmetric.json test/test_diffractive_surfaces.py
git commit -m "feat(diffractive): add RotationallySymmetric achromat DOE (Dun et al. Optica 2020)"
```

---

### Task 4: Verification script

**Files:**
- Create: `9_diffractive_surfaces.py` (repo root, following the `0_hello_*` / `8_*` numbered-script convention)

- [ ] **Step 1: Write the script** `9_diffractive_surfaces.py`:

```python
"""Demonstrate the three paper-based diffractive surfaces and their signature PSFs.

  * Rank1 (Sun et al., CVPR 2020): cross / streak PSF for HDR.
  * DiffractedRotation (Jeon et al., TOG 2019): PSF rotates with wavelength.
  * RotationallySymmetric (Dun et al., Optica 2020): rotationally-symmetric PSF.

Runs on GPU when available. PSFs are saved under ./outputs/diffractive_surfaces/.
"""

import math
import os

import torch
from torchvision.utils import save_image

from deeplens import DiffractiveLens
from deeplens.diffractive_surface import Rank1

OUT = "./outputs/diffractive_surfaces"
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def demo_rank1():
    """Saddle-initialized rank-1 DOE -> X/cross-shaped PSF."""
    lens = DiffractiveLens(
        filename="./datasets/lenses/diffraclens/rank1.json",
        dtype=torch.float64,
        device=DEVICE,
    )
    r1 = [s for s in lens.surfaces if isinstance(s, Rank1)][0]
    n0, n1 = r1.res
    ramp0 = torch.linspace(-4, 4, n0, device=lens.device, dtype=torch.float64)
    ramp1 = torch.linspace(-4, 4, n1, device=lens.device, dtype=torch.float64)
    r1.V = ramp0[:, None]            # rank-1 saddle: V @ Q.T = outer(ramp0, ramp1)
    r1.Q = ramp1[:, None]
    psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=128)
    save_image(psf[None].clamp(min=0), f"{OUT}/rank1_psf.png", normalize=True)
    print(f"[Rank1] PSF saved, sum={psf.sum():.3f}")


def _principal_angle(psf):
    """Orientation (deg) of the PSF's principal axis via the intensity covariance."""
    h, w = psf.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=psf.device, dtype=psf.dtype),
        torch.arange(w, device=psf.device, dtype=psf.dtype),
        indexing="ij",
    )
    p = psf / psf.sum()
    cy, cx = (p * yy).sum(), (p * xx).sum()
    yy, xx = yy - cy, xx - cx
    cxx = (p * xx * xx).sum()
    cyy = (p * yy * yy).sum()
    cxy = (p * xx * yy).sum()
    return math.degrees(0.5 * math.atan2(2 * cxy.item(), (cxx - cyy).item()))


def demo_diffracted_rotation():
    """Sweep wavelength; PSF lobe orientation should rotate monotonically."""
    lens = DiffractiveLens(
        filename="./datasets/lenses/diffraclens/diffracted_rotation.json",
        dtype=torch.float64,
        device=DEVICE,
    )
    wvlns = [0.45, 0.50, 0.55, 0.60, 0.65]
    frames = []
    for wvln in wvlns:
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=128, wvln=wvln)
        frames.append(psf.clamp(min=0))
        print(f"[DiffractedRotation] wvln={wvln:.2f}um  angle={_principal_angle(psf):+.1f} deg")
    montage = torch.stack(frames, dim=0)[:, None]  # [n, 1, ks, ks]
    save_image(montage, f"{OUT}/diffracted_rotation_sweep.png", nrow=len(wvlns), normalize=True)
    print("[DiffractedRotation] wavelength sweep montage saved")


def demo_rotational_symmetric():
    """Rotationally-symmetric PSF at several wavelengths."""
    lens = DiffractiveLens(
        filename="./datasets/lenses/diffraclens/rotational_symmetric.json",
        dtype=torch.float64,
        device=DEVICE,
    )
    frames = []
    for wvln in [0.45, 0.55, 0.65]:
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=128, wvln=wvln)
        frames.append(psf.clamp(min=0))
        print(f"[RotationallySymmetric] wvln={wvln:.2f}um  sum={psf.sum():.3f}")
    montage = torch.stack(frames, dim=0)[:, None]
    save_image(montage, f"{OUT}/rotational_symmetric_psf.png", nrow=len(frames), normalize=True)
    print("[RotationallySymmetric] multi-wavelength PSF saved "
          "(achromaticity requires end-to-end training, out of scope)")


if __name__ == "__main__":
    demo_rank1()
    demo_diffracted_rotation()
    demo_rotational_symmetric()
    print(f"\nDone. Images in {OUT}/")
```

- [ ] **Step 2: Run locally (CPU smoke test, small)** — run `python 9_diffractive_surfaces.py`. Expected: prints per-surface lines, writes 3 images to `outputs/diffractive_surfaces/`, no exceptions. (CPU is slow for 1000² FFTs; if too slow locally, defer the full run to GPU in Task 5.)

- [ ] **Step 3: Inspect the angle trend** — confirm `[DiffractedRotation]` printed angles change monotonically (modulo ±90° wrap) across the wavelength sweep.

- [ ] **Step 4: Commit**

```bash
git add 9_diffractive_surfaces.py
git commit -m "feat(diffractive): add verification script for the three new DOE surfaces"
```

---

### Task 5: GPU verification on AutoDL

**REQUIRED SUB-SKILL:** Use `autodl-runner` to sync this branch and run on the GPU machine.

- [ ] **Step 1:** Sync the worktree to AutoDL (per the `autodl-runner` workflow / `reference_autodl_deeplens` memory).
- [ ] **Step 2:** Run `pytest test/test_diffractive_surfaces.py -q` on the GPU. Expected: all pass.
- [ ] **Step 3:** Run `python 9_diffractive_surfaces.py` on the GPU. Expected: 3 images produced; DiffractedRotation angles monotonic across the sweep.
- [ ] **Step 4:** Pull back / inspect the PSF images and confirm: Rank1 shows an X/cross; DiffractedRotation lobe rotates with wavelength; RotationallySymmetric PSF is rotationally symmetric.

---

## Self-Review

**Spec coverage:** Rank1 (Task 1), DiffractedRotation (Task 2), RotationallySymmetric (Task 3), loader+configs (Tasks 1-3), tests incl. property tests (Tasks 1-3), verification script (Task 4), GPU verification (Task 5). All spec sections covered.

**Placeholder scan:** No TBD/TODO; every code step has complete code.

**Type consistency:** `phase_func`/`init_from_dict`/`get_optimizer_params`/`surf_dict` signatures match the base class and existing surfaces. `write_lens_json` path tuple grows in Task 1 (`Pixel2D, Rank1`) then Task 3 (`+ RotationallySymmetric`) — consistent. `DiffractedRotation.surf_dict()` takes no path (scalar params only); `Rank1`/`RotationallySymmetric` take a `weight_path`. Loader type keys are lowercased: `rank1`, `diffractedrotation`, `rotationallysymmetric`.
