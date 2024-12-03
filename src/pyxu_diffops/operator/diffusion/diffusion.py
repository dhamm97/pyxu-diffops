import collections.abc as cabc
import typing as typ

import numpy as np
import pyxu.info.ptype as pyct
import pyxu.operator.linop.diff as pydiff
import pyxu.operator.linop.filter as pyfilt
import pyxu.util as pycu

from pyxu_diffops.operator.diffusion._diffusion import _Diffusion
from pyxu_diffops.operator.diffusion._diffusion_coeff import (
    DiffusionCoeffAnisoCoherenceEnhancing,
    DiffusionCoeffAnisoEdgeEnhancing,
    DiffusionCoeffAnisotropic,
    DiffusionCoeffIsotropic,
    _DiffusionCoefficient,
)
from pyxu_diffops.operator.diffusion._diffusivity import (
    MfiDiffusivity,
    PeronaMalikDiffusivity,
    TotalVariationDiffusivity,
)
from pyxu_diffops.operator.diffusion._extra_diffusion_term import (
    CurvaturePreservingTerm,
    MfiExtraTerm,
)

__all__ = [
    "MfiDiffusion",
    "PeronaMalikDiffusion",
    "TikhonovDiffusion",
    "TotalVariationDiffusion",
    "CurvaturePreservingDiffusionOp",
    "AnisEdgeEnhancingDiffusionOp",
    "AnisCoherenceEnhancingDiffusionOp",
    "AnisDiffusionOp",
]
# "AnisMfiDiffusionOp"]

# pxa.LinOp


class MfiDiffusion(_Diffusion):
    r"""
    Minimum Fisher Information (MFI) diffusion operator, featuring an inhomogeneous isotropic diffusion tensor.
    Inspired from minimally informative prior principles, it is popular in the plasma physics tomography community.
    The diffusivity decreases for increasing local intensity values, to preserve bright features.

    The MFI diffusion tensor is defined, for the :math:`i`-th pixel, as:

    .. math::
        \big(\mathbf{D}(\mathbf{f})\big)_i = (g(\mathbf{f}))_i \;\mathbf{I} = \begin{pmatrix}(g(\mathbf{f}))_i & 0\\ 0 & (g(\mathbf{f}))_i\end{pmatrix},

    where the diffusivity :math:`g` is defined in two possible ways:

    * in the *tame* case as

      .. math::
         (g(\mathbf{f}))_i = \frac{1}{1+\max\{\delta, f_i\}/\beta},\quad\delta\ll 1,
    :math:`\quad\;\;` where :math:`\beta` is a contrast parameter,

    * in the *untame* case as

      .. math::
         (g(\mathbf{f}))_i = \frac{1}{\max\{\delta, f_i\}},\quad\delta\ll 1,

    The gradient of the operator reads

    .. math::
       -\mathrm{div}(\mathbf{D}(\mathbf{f})\boldsymbol{\nabla}\mathbf{f})-\sum_{i=0}^{N_{tot}-1}\frac{\vert(\boldsymbol{\nabla}\mathbf{f})_i\vert^2}{(g(\mathbf{f}))_i^2}\;,

    where the sum is an optional balloon force `extra_term`, which can be included or not.

    We recommend using the *tame* version, since it is better behaved: the *untame* version requires very small
    steps to be stable. Furthermore, we recommend including the extra term, because in this case the
    potential :math:`\phi` can be defined.

    Parameters
    ----------
    dim_shape: NDArrayShape
        Shape of the input array.
    extra_term: bool
        Whether the extra term arising from the differentiation of the MFI functional should be
        included. Defaults to `True` (recommended). If set to `False` (*linearized MFI*), the MFI diffusion operator
        does not admit a potential and the ``apply()`` method is not defined.
    beta: Real
        Contrast parameter, determines the magnitude above which image gradients are preserved. Defaults to 1.
    clipping_value: Real
        Clipping value :math:`\delta` in the expression of the MFI diffusivity. Defaults to 1e-5.
    tame: bool
       Whether tame or untame version should be used. Defaults to `True` (tame, recommended).
    sampling: Real, list[Real]
            Sampling step (i.e. distance between two consecutive elements of an array).
            Defaults to 1.

    Returns
    -------
    op: OpT
            MFI diffusion operator.

    """

    def __init__(
        self,
        dim_shape: pyct.NDArrayShape,
        extra_term: bool = True,
        beta: pyct.Real = 1,
        clipping_value: pyct.Real = 1e-5,
        tame: bool = True,
        sampling: typ.Union[pyct.Real, cabc.Sequence[pyct.Real, ...]] = 1,
    ):
        gradient = pydiff.Gradient(
            dim_shape=dim_shape,
            directions=(1, 2),
            diff_method="fd",
            scheme="forward",
            mode="symmetric",
            sampling=sampling,
        )
        mfi_diffusion_coeff = DiffusionCoeffIsotropic(
            dim_shape=dim_shape,
            diffusivity=MfiDiffusivity(dim_shape=dim_shape, beta=beta, clipping_value=clipping_value, tame=tame),
        )
        if extra_term:
            mfi_extra_term = MfiExtraTerm(
                dim_shape=dim_shape, gradient=gradient, beta=beta, clipping_value=clipping_value, tame=tame
            )
            super().__init__(
                dim_shape,
                gradient=gradient,
                diffusion_coefficient=mfi_diffusion_coeff,
                extra_diffusion_term=mfi_extra_term,
            )
        else:
            super().__init__(dim_shape, gradient=gradient, diffusion_coefficient=mfi_diffusion_coeff)
        # initialize lipschitz constants. should this be done instead inside _DiffusionOp.__init__ ??
        if tame:
            self.diff_lipschitz = gradient.lipschitz**2
        else:
            self.diff_lipschitz = gradient.lipschitz**2 / clipping_value
        # REMARK: I think that with extra term we don't have lipschitzianity, only Holder-continuity order 2.
        # However, I did not observe instability in practice. I think it's due to denominator or order norm(x)!
        # if that's true, a reasonable estimate would be the following (doubling diff_lipschitz)
        if extra_term:
            self.diff_lipschitz *= 2

    def apply(self, arr: pyct.NDArray) -> pyct.Real:
        if self.extra_diffusion_term is not None:
            xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
            y = self._compute_divergence_term(arr)  # (batch,nchannels,nx,ny)
            return xp.einsum("...jkl,...jkl->...", y, arr) / 2  # (batch,)
        else:
            raise RuntimeError(
                "'apply()' method not defined for MfiDiffusionOp with no MfiExtraTerm: no underlying variational intepretation"
            )


class PeronaMalikDiffusion(_Diffusion):
    r"""
    Perona-Malik diffusion operator, featuring an inhomogeneous isotropic diffusion tensor with Perona-Malik diffusivity. It can be used for edge-preserving smoothing.
    It reduces the diffusion intensity in locations characterised by large gradients, effectively achieving edge-preservation.

    The gradient of the operator reads :math:`\nabla\phi(\mathbf{f})=-\mathrm{div}(\mathbf{D}(\mathbf{f})\boldsymbol{\nabla}\mathbf{f})`,
    where :math:`\phi` is the potential.

    The Perona-Malik diffusion tensor is defined, for the :math:`i`-th pixel, as:

    .. math::

        \big(\mathbf{D}(\mathbf{f})\big)_i = (g(\mathbf{f}))_i \;\mathbf{I} = \begin{pmatrix}(g(\mathbf{f}))_i & 0\\ 0 & (g(\mathbf{f}))_i\end{pmatrix},

    where the diffusivity with contrast parameter :math:`\beta` is defined as:

    * :math:`(g(\mathbf{f}))_i = \exp(-\vert (\nabla_\sigma \mathbf{f})_i \vert ^2 / \beta^2)` in the exponential case,

    * :math:`(g(\mathbf{f}))_i = 1/\big( 1+\vert (\nabla_\sigma \mathbf{f})_i \vert ^2 / \beta^2\big)` in the rational case.

    Gaussian derivatives with width :math:`\sigma` are used for the gradient as customary with the ill-posed Perona-Malik diffusion process.

    In both cases, the corresponding divergence-based diffusion term admits a potential
    (see [Tschumperle]_ for the exponential case): the ``apply()`` method is well-defined.

    Parameters
    ----------
    dim_shape: NDArrayShape
        Shape of the input array.
    beta: Real
        Contrast parameter, determines the magnitude above which image gradients are preserved. Defaults to 1.
    pm_fct: str
        Perona-Malik diffusivity function. Must be either 'exponential' or 'rational'.
    sigma_gd: Real
        Standard deviation for kernel of Gaussian derivatives used for gradient computation. Defaults to 1.
    sampling: Real, list[Real]
            Sampling step (i.e. distance between two consecutive elements of an array).
            Defaults to 1.

    Returns
    -------
    op: OpT
            Perona-Malik diffusion operator.

    Example
    -------

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        import pyxu.opt.solver as pysol
        import pyxu.abc.solver as pysolver
        import pyxu.opt.stop as pystop
        import pyxu_diffops.operator as pyxop
        import skimage as skim

        # Import RGB image
        image = skim.data.cat().astype(float)
        print(image.shape)  # (300, 451, 3)

        # Move color-stacking axis to front (needed for pyxu stacking convention)
        image = np.moveaxis(image, 2, 0)
        print(image.shape)  # (3, 300, 451)

        # Instantiate diffusion operator
        pm_diffop = pyxop.PeronaMalikDiffusion(dim_shape=(3, 300, 451), beta=5)

        # Define PGD solver, with stopping criterion and starting point x0
        stop_crit = pystop.MaxIter(n=100)

        # Perform 50 gradient flow iterations starting from x0
        PGD = pysol.PGD(f=pm_diffop, show_progress=False, verbosity=100)
        PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=image, stop_crit=stop_crit,
                       tau=2 / pm_diffop.diff_lipschitz))
        pm_smoothed_image = PGD.solution()

        # Reshape images for plotting
        image = np.moveaxis(image, 0, 2)
        pm_smoothed_image = np.moveaxis(pm_smoothed_image, 0, 2)

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image.astype(int))
        ax[0].set_title("Image", fontsize=15)
        ax[0].axis('off')
        ax[1].imshow(pm_smoothed_image.astype(int))
        ax[1].set_title("100 iterations Perona Malik diffusion", fontsize=15)
        ax[1].axis('off')

    """

    def __init__(
        self,
        dim_shape: pyct.NDArrayShape,
        beta: pyct.Real = 1,
        pm_fct: str = "exponential",
        sigma_gd: pyct.Real = 1,
        sampling: typ.Union[pyct.Real, cabc.Sequence[pyct.Real, ...]] = 1,
    ):
        gradient = pydiff.Gradient(
            dim_shape=dim_shape,
            directions=(1, 2),
            diff_method="fd",
            scheme="forward",
            mode="symmetric",
            sampling=sampling,
        )
        gaussian_gradient = pydiff.Gradient(
            dim_shape=dim_shape, directions=(1, 2), diff_method="gd", sigma=sigma_gd, sampling=sampling
        )
        pm_diffusion_coeff = DiffusionCoeffIsotropic(
            dim_shape=dim_shape,
            diffusivity=PeronaMalikDiffusivity(
                dim_shape=dim_shape, gradient=gaussian_gradient, beta=beta, pm_fct=pm_fct
            ),
        )
        super().__init__(dim_shape, gradient=gradient, diffusion_coefficient=pm_diffusion_coeff)
        self.beta = beta
        self.pm_fct = pm_fct
        # initialize lipschitz constants. should this be done instead inside _DiffusionOp.__init__ ??
        self.diff_lipschitz = gradient.lipschitz**2

    def _apply_exponential(self, arr: pyct.NDArray) -> pyct.Real:
        xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
        # Inplace implementation of
        #     0.5*(beta**2)*sum(1 - xp.exp(-grad_norm_sq/beta**2))
        y = -self.diffusion_coefficient.diffusivity._compute_grad_norm_sq(arr, self.gradient)  # (batch,nchannels,nx,ny)
        y /= self.beta**2
        y = 1 - xp.exp(y)
        z = xp.sum(y, axis=(-3, -2, -1))  # (batch,)
        z *= self.beta**2
        return 0.5 * z  # (batch,)

    def _apply_rational(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
        # Inplace implementation of
        #   0.5*(beta**2)*sum(xp.log(1+grad_norm_sq/beta**2)
        y = self.diffusion_coefficient.diffusivity._compute_grad_norm_sq(arr, self.gradient)  # (batch,nchannels,nx,ny)
        y /= self.beta**2
        y += 1
        y = xp.log(y)
        z = xp.sum(y, axis=(-3, -2, -1))  # (batch,)
        z *= self.beta**2
        return 0.5 * z  # (batch,)

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        f = dict(
            exponential=self._apply_exponential,
            rational=self._apply_rational,
        ).get(self.pm_fct)
        out = f(arr)
        return out


class TikhonovDiffusion(_Diffusion):
    r"""
    Tikhonov diffusion operator, featuring a homogeneous isotropic diffusion tensor with constant diffusivity. It
    achieves Gaussian smoothing, isotropically smoothing in all directions. The diffusion intensity is identical at all
    locations. Smoothing with this diffusion operator blurs the original image.

    The gradient of the operator reads :math:`\nabla\phi(\mathbf{f})=-\mathrm{div}(\boldsymbol{\nabla}\mathbf{f})=-\boldsymbol{\Delta}\mathbf{f}`;
    it derives from the potential :math:`\phi=\Vert\boldsymbol{\nabla}\mathbf{f}\Vert_2^2`.

    The Tikohnov diffusion tensor is defined, for the :math:`i`-th pixel, as:

    .. math::

        \big(\mathbf{D}(\mathbf{f})\big)_i = \mathbf{I} = \begin{pmatrix}1 & 0\\ 0 & 1\end{pmatrix}.

    Parameters
    ----------
    dim_shape: NDArrayShape
        Shape of the input array.
    sampling: Real, list[Real]
            Sampling step (i.e. distance between two consecutive elements of an array).
            Defaults to 1.

    Returns
    -------
    op: OpT
            Tikhonov diffusion operator.

    Example
    -------

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        import pyxu.opt.solver as pysol
        import pyxu.abc.solver as pysolver
        import pyxu.opt.stop as pystop
        import pyxu_diffops.operator as pyxop
        import skimage as skim

        # Import RGB image
        image = skim.data.cat().astype(float)
        print(image.shape)  # (300, 451, 3)

        # Move color-stacking axis to front (needed for pyxu stacking convention)
        image = np.moveaxis(image, 2, 0)
        print(image.shape)  # (3, 300, 451)

        # Instantiate diffusion operator
        tikh_diffop = pyxop.TikhonovDiffusion(dim_shape=(3, 300, 451))

        # Define PGD solver, with stopping criterion and starting point x0
        stop_crit = pystop.MaxIter(n=100)

        # Perform 50 gradient flow iterations starting from x0
        PGD = pysol.PGD(f=tikh_diffop, show_progress=False, verbosity=100)
        PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=image, stop_crit=stop_crit,
                       tau=2 / tikh_diffop.diff_lipschitz))
        tikh_smoothed_image = PGD.solution()

        # Reshape images for plotting.
        image = np.moveaxis(image, 0, 2)
        tikh_smoothed_image = np.moveaxis(tikh_smoothed_image, 0, 2)

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image.astype(int))
        ax[0].set_title("Image", fontsize=15)
        ax[0].axis('off')
        ax[1].imshow(tikh_smoothed_image.astype(int))
        ax[1].set_title("100 iterations Tikhonov smoothing", fontsize=15)
        ax[1].axis('off')

    """

    def __init__(
        self,
        dim_shape: pyct.NDArrayShape,
        sampling: typ.Union[pyct.Real, cabc.Sequence[pyct.Real, ...]] = 1,
    ):
        gradient = pydiff.Gradient(
            dim_shape=dim_shape,
            directions=(1, 2),
            diff_method="fd",
            scheme="forward",
            mode="symmetric",
            sampling=sampling,
        )
        tikhonov_diffusion_coeff = DiffusionCoeffIsotropic(dim_shape=dim_shape)
        super().__init__(dim_shape, gradient=gradient, diffusion_coefficient=tikhonov_diffusion_coeff)
        # initialize lipschitz constants. should this be done instead inside _DiffusionOp.__init__ ??
        self.diff_lipschitz = gradient.lipschitz**2

    def apply(self, arr: pyct.NDArray) -> pyct.Real:
        xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
        y = self._compute_divergence_term(arr)  # (batch,nchannels,nx,ny)
        return xp.einsum("...jkl,...jkl->...", y, arr) / 2  # (batch,)


class TotalVariationDiffusion(_Diffusion):
    r"""
    Total variation (TV) diffusion operator, featuring an inhomogeneous isotropic diffusion tensor. It can be used for edge-enhancing smoothing.
    It reduces the diffusion intensity in locations characterised by large gradients, effectively achieving edge-preservation. The
    diffusion operator formulation of total variation regularization stems from the Euler-Lagrange equations associated to the
    problem of minimizing the TV regularization functional.

    The gradient of the operator reads :math:`\nabla\phi(\mathbf{f})=-\mathrm{div}(\mathbf{D}(\mathbf{f})\boldsymbol{\nabla}\mathbf{f})`,
    deriving from the total variation potential :math:`\phi`.
    In the TV classical (*untame*) formulation, it holds
    :math:`\phi=\Vert \boldsymbol{\nabla}\mathbf{f}\Vert_{2,1}=\sum_{i=0}^{N_{tot}-1}\vert(\boldsymbol{\nabla}\mathbf{f})_i\vert`.
    A similar definition holds for its *tame* version (recommended).

    The TV diffusion tensor is defined, for the :math:`i`-th pixel, as:

    .. math::

        \big(\mathbf{D}(\mathbf{f})\big)_i = (g(\mathbf{f}))_i \;\mathbf{I} = \begin{pmatrix}(g(\mathbf{f}))_i & 0\\ 0 & (g(\mathbf{f}))_i\end{pmatrix}.

    The diffusivity can be defined in two possible ways:

    * in the *tame* case as
    .. math ::
        (g(\mathbf{f}))_i = \frac{\beta} { \sqrt{\beta^2+ \vert (\boldsymbol{\nabla} \mathbf{f})_i \vert ^2}}, \quad \forall i

    * in the *untame* case as
    .. math ::
        (g(\mathbf{f}))_i = \frac{1} { \vert (\boldsymbol{\nabla} \mathbf{f})_i \vert}, \quad \forall i.

    The *tame* formulation amounts to an approximation of the TV functional very similar to the Huber loss approach. The
    parameter :math:`\beta` controls the quality of the smooth approximation of the :math:`L^2` norm
    :math:`\vert (\boldsymbol{\nabla} \mathbf{f})_i \vert` involved in the TV
    approach. Lower values correspond to better approximations but typically lead to larger computational cost.

    We recommended using the *tame* version; the *untame* one is unstable.

    Parameters
    ----------
    dim_shape: NDArrayShape
        Shape of the input array.
    beta: Real
        Contrast parameter, determines the magnitude above which image gradients are preserved. Defaults to 1.
    tame: bool
       Whether tame or untame version should be used. Defaults to `True` (tame, recommended).
    sampling: Real, list[Real]
            Sampling step (i.e. distance between two consecutive elements of an array).
            Defaults to 1.

    Returns
    -------
    op: OpT
            Total variation diffusion operator.

    Example
    -------

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        import pyxu.opt.solver as pysol
        import pyxu.abc.solver as pysolver
        import pyxu.opt.stop as pystop
        import pyxu_diffops.operator as pyxop
        import skimage as skim

        # Import RGB image
        image = skim.data.cat().astype(float)
        print(image.shape)  # (300, 451, 3)

        # Move color-stacking axis to front (needed for pyxu stacking convention)
        image = np.moveaxis(image, 2, 0)
        print(image.shape)  # (3, 300, 451)

        # Instantiate diffusion operator
        tv_diffop = pyxop.TotalVariationDiffusion(dim_shape=(3, 300, 451), beta=2)

        # Define PGD solver, with stopping criterion and starting point x0
        stop_crit = pystop.MaxIter(n=100)

        # Perform 50 gradient flow iterations starting from x0
        PGD = pysol.PGD(f=tv_diffop, show_progress=False, verbosity=100)
        PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=image, stop_crit=stop_crit,
                       tau=2/tv_diffop.diff_lipschitz))
        tv_smoothed_image = PGD.solution()

        # Reshape images for plotting.
        image = np.moveaxis(image, 0, 2)
        tv_smoothed_image = np.moveaxis(tv_smoothed_image, 0, 2)

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image.astype(int))
        ax[0].set_title("Image", fontsize=15)
        ax[0].axis('off')
        ax[1].imshow(tv_smoothed_image.astype(int))
        ax[1].set_title("100 iterations Total Variation smoothing", fontsize=15)
        ax[1].axis('off')

    """

    def __init__(
        self,
        dim_shape: pyct.NDArrayShape,
        beta: pyct.Real = 1,
        tame: bool = True,
        # sigma_gd: pyct.Real = 1,
        sampling: typ.Union[pyct.Real, cabc.Sequence[pyct.Real, ...]] = 1,
    ):
        gradient = pydiff.Gradient(
            dim_shape=dim_shape,
            directions=(1, 2),
            diff_method="fd",
            scheme="forward",
            mode="symmetric",
            sampling=sampling,
        )
        tv_diffusion_coeff = DiffusionCoeffIsotropic(
            dim_shape=dim_shape,
            diffusivity=TotalVariationDiffusivity(dim_shape=dim_shape, gradient=gradient, beta=beta, tame=tame),
        )
        super().__init__(dim_shape, gradient=gradient, diffusion_coefficient=tv_diffusion_coeff)
        self.beta = beta
        self.tame = tame
        # initialize lipschitz constants. should this be done instead inside _DiffusionOp.__init__ ??
        self.diff_lipschitz = gradient.lipschitz**2
        if not self.tame:
            self.diff_lipschitz = np.inf

    # @pycrt.enforce_precision(i="arr")
    def _apply_tame(self, arr: pyct.NDArray) -> pyct.NDArray:
        # Inplace implementation of
        #   beta**2*sum(xp.sqrt(1+grad_norm_sq/beta**2)
        xp = pycu.get_array_module(arr)
        y = self.diffusion_coefficient.diffusivity._compute_grad_norm_sq(arr, self.gradient)  # (batch,nchannels,nx,ny)
        y = y[..., 0, :, :]  # (batch,nx,ny) all channels are the same, duplicated information
        y /= self.beta**2
        y += 1
        y = xp.sqrt(y)
        z = xp.sum(y, axis=(-2, -1))  # (batch,)
        return self.beta**2 * z

    # @pycrt.enforce_precision(i="arr")
    def _apply_untamed(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = self.diffusion_coefficient.diffusivity._compute_grad_norm_sq(arr, self.gradient)  # (batch,nchannels,nx,ny)
        y = xp.sqrt(y[..., 0, :, :])  # (batch,nx,ny)
        return xp.sum(y, axis=(-2, -1))  # (batch,)

    # @pycrt.enforce_precision(i="arr")
    # @pycu.vectorize("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        f = self._apply_tame if self.tame else self._apply_untamed
        out = f(arr)
        return out


class CurvaturePreservingDiffusionOp(_Diffusion):
    r"""
    Curvature preserving diffusion operator [Tschumperle]_. This trace-based operator promotes curvature
    preservation along a given vector field :math:`\mathbf{w}`, defined by a :math:`2`-dimensional
    vector :math:`\mathbf{w}_i\in\mathbb{R}^2` for each pixel :math:`i`.

    The gradient of the operator reads

    .. math::
        \frac{\partial\mathbf{f}}{\partial t} = -\mathrm{trace}\big(\mathbf{T}(\mathbf{w})\mathbf{H}(\mathbf{f})\big) - (\nabla\mathbf{f})^T \mathbf{J}_{\mathbf{w}}\mathbf{w},

    where the trace diffusion tensor :math:`\mathbf{T}(\mathbf{f})` is defined, for the :math:`i`-th pixel, as

    .. math::
        \big(\mathbf{T}(\mathbf{f})\big)_i=\mathbf{w}_i\mathbf{w}_i^T.

    The curvature preserving diffusion operator does not admit a potential.

    Parameters
    ----------
    dim_shape: NDArrayShape
        Shape of the input array.
    curvature_preservation_field: NDArray
        Vector field of shape :math:`(2, N_0, N_1)` along which curvature should be preserved.
    sampling: Real, list[Real]
            Sampling step (i.e. distance between two consecutive elements of an array).
            Defaults to 1.

    Returns
    -------
    op: OpT
            Curvature preserving diffusion operator.

    Example
    -------

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        import pyxu.operator.linop.diff as pydiff
        import pyxu.opt.solver as pysol
        import pyxu.abc.solver as pysolver
        import pyxu.opt.stop as pystop
        import pyxu_diffops.operator as pyxop
        import skimage as skim

        # Define random image
        image = 255 * np.random.rand(3, 300, 451)

        # Define vector field, diffusion process will preserve curvature along it
        image_center = np.array(image.shape[1:]) / 2 + [0.25, 0.25]
        curvature_preservation_field = np.zeros((2, np.prod(image.shape[1:])))
        curv_pres_1 = np.zeros(image.shape[1:])
        curv_pres_2 = np.zeros(image.shape[1:])
        for i in range(image.shape[1]):
            for j in range(image.shape[2]):
                theta = np.arctan2(-i + image_center[0], j - image_center[1])
                curv_pres_1[i, j] = np.cos(theta)
                curv_pres_2[i, j] = np.sin(theta)
        curvature_preservation_field[0, :] = curv_pres_1.reshape(1, -1)
        curvature_preservation_field[1, :] = curv_pres_2.reshape(1, -1)
        curvature_preservation_field = curvature_preservation_field.reshape(2, *image.shape[1:])

        # Define curvature-preserving diffusion operator
        CurvPresDiffusionOp = pyxop.CurvaturePreservingDiffusionOp(dim_shape=(3, 300, 451),
                                                                    curvature_preservation_field=curvature_preservation_field)

        # Perform 500 gradient flow iterations
        stop_crit = pystop.MaxIter(n=200)
        PGD_curve = pysol.PGD(f=CurvPresDiffusionOp, g=None, show_progress=False, verbosity=100)

        PGD_curve.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=image, stop_crit=stop_crit, acceleration=False,
                             tau=1 / CurvPresDiffusionOp.diff_lipschitz))
        curv_smooth_image = PGD_curve.solution()

        # Reshape images for plotting.
        image = np.moveaxis(image, 0, 2)
        curv_smooth_image = np.moveaxis(curv_smooth_image, 0, 2)
        # Rescale for better constrast
        curv_smooth_image -= np.min(curv_smooth_image)
        curv_smooth_image /= np.max(curv_smooth_image)
        curv_smooth_image *= 255

        # Plot
        fig, ax = plt.subplots(1, 3, figsize=(30, 6))
        ax[0].imshow(image.astype(int), cmap="gray", aspect="auto")
        ax[0].set_title("Image", fontsize=30)
        ax[0].axis('off')
        ax[1].quiver(curv_pres_2[::40, ::60], curv_pres_1[::40, ::60])
        ax[1].set_title("Vector field", fontsize=30)
        ax[1].axis('off')
        ax[2].imshow(curv_smooth_image.astype(int), cmap="gray", aspect="auto")
        ax[2].set_title("200 iterations Curvature Preserving", fontsize=30)
        ax[2].axis('off')

    """

    def __init__(
        self,
        dim_shape: pyct.NDArrayShape,
        curvature_preservation_field: pyct.NDArray = None,  # (2dim,nx,ny)
        sampling: typ.Union[pyct.Real, cabc.Sequence[pyct.Real, ...]] = 1,
    ):
        xp = pycu.get_array_module(curvature_preservation_field)
        self.nchannels = dim_shape[0]
        gradient = pydiff.Gradient(
            dim_shape=dim_shape, directions=(1, 2), diff_method="fd", scheme="central", mode="edge", sampling=sampling
        )
        hessian = pydiff.Hessian(
            dim_shape=dim_shape[1:], diff_method="fd", mode="symmetric", sampling=sampling, scheme="central", accuracy=2
        )
        curvature_preserving_term = CurvaturePreservingTerm(
            dim_shape=dim_shape, gradient=gradient, curvature_preservation_field=curvature_preservation_field
        )
        # assemble trace diffusion tensor
        cf_sh = curvature_preservation_field.shape
        tensors = curvature_preservation_field.reshape(1, *cf_sh) * curvature_preservation_field.reshape(
            cf_sh[0], 1, *cf_sh[1:]
        )  # (2dim,2dim,nx,ny)
        tensors = xp.stack([tensors] * self.nchannels, -3)  # (2dim,2dim,nchannels,nx,ny)
        trace_diffusion_coefficient = _DiffusionCoefficient(dim_shape=dim_shape, isotropic=False, trace_term=True)
        trace_diffusion_coefficient.set_frozen_coeff(tensors)  # (2dim,2dim,nchannels,nx,ny)

        super().__init__(
            dim_shape,
            hessian=hessian,
            trace_diffusion_coefficient=trace_diffusion_coefficient,
            extra_diffusion_term=curvature_preserving_term,
        )
        # initialize lipschitz constants. should this be done instead inside _DiffusionOp.__init__ ? I think it's fine here.
        self.diff_lipschitz = (
            hessian.lipschitz * curvature_preserving_term.max_norm + curvature_preserving_term.lipschitz
        )


class AnisEdgeEnhancingDiffusionOp(_Diffusion):
    r"""
    Anisotropic edge-enhancing diffusion operator, featuring an inhomogeneous anisotropic diffusion tensor
    defined as in [Weickert]_. The diffusion tensor is defined as a function of the
    :py:class:`~pyxu.operator.linop.filter.StructureTensor`,
    which is a tensor describing the properties of the image in the neighbourhood of each pixel.
    This diffusion operator allows edge enhancement by reducing the flux in the direction of the
    eigenvector of the structure tensor with largest eigenvalue. Essentially, smoothing across
    the detected edges is inhibited in a more sophisticated way compared to isotropic approaches,
    by considering not only the local gradient but its behaviour in a neighbourhood of the pixel.

    The gradient of the operator reads :math:`-\mathrm{div}(\mathbf{D}(\mathbf{f})\boldsymbol{\nabla}\mathbf{f})`.

    Let :math:`\mathbf{v}_0^i,\mathbf{v}_1^i` be the eigenvectors of the :math:`i`-th pixel structure tensor
    :math:`S_i`, with eigenvalues :math:`e_0^i, e_1^i` sorted in decreasing order. The diffusion tensor is defined,
    for the :math:`i`-th pixel, as the symmetric positive definite tensor:

    .. math::
        \big(\mathbf{D}(\mathbf{f})\big)_i = \lambda_0^i\mathbf{v}_0^i(\mathbf{v}_0^i)^T+\lambda_1^i\mathbf{v}_1^i(\mathbf{v}_1^i)^T,


    where the smoothing intensities along the two eigendirections are defined, to achieve edge-enhancing, as

    .. math::
        \lambda_0^i &= g(e_0^i) :=
       \begin{cases}
           1 & \text{if } e_0^i = 0 \\
           1 - \exp\big(\frac{-C}{(e_0^i/\beta)^m}\big) & \text{if } e_0^i >0,
       \end{cases}\\
        \lambda_1^i &= 1,

    where :math:`\beta` is a contrast parameter, :math:`m` controls the decay rate of :math:`\lambda_0^i` as a function
    of :math:`e_0^i`, and :math:`C\in\mathbb{R}` is a normalization constant (see [Weickert]_).

    Since :math:`\lambda_1^i \geq \lambda_0^i`, the smoothing intensity is stronger in the direction of the second
    eigenvector of :math:`\mathbf{S}_i`, and therefore perpendicular to the (locally averaged) gradient.
    Moreover, :math:`\lambda_0^i` is a decreasing function of :math:`e_0^i`, which indicates that when the gradient
    magnitude is high (sharp edges), smoothing in the direction of the gradient is inhibited, i.e., edges are preserved.

    In general, the diffusion operator does not admit a potential. However, if the diffusion tensor is evaluated
    at some image :math:`\tilde{\mathbf{f}}` and kept fixed to :math:`\mathbf{D}(\tilde{\mathbf{f}})`, the operator
    derives from the potential :math:`\Vert\sqrt{\mathbf{D}(\tilde{\mathbf{f}})}\boldsymbol{\nabla}\mathbf{f}\Vert_2^2`.
    By doing so, the smoothing directions and intensities are fixed according to the features of an image of interest.
    This can be achieved passing a *freezing array* at initialization. In such cases, a *matrix-based implementation*
    is recommended for efficiency.

    Parameters
    ----------
    dim_shape: NDArrayShape
        Shape of the input array.
    beta: Real
        Contrast parameter, determines the gradient magnitude for edge enhancement. Defaults to 1.
    m: Real
        Decay parameter, determines how quickly the smoothing effect changes as a function of :math:`e_0^i/\beta`. Defaults to 4.
    sigma_gd_st: Real
       Gaussian width of the gaussian derivative involved in the structure tensor computation. Defaults to 2.
    smooth_sigma_st: Real
       Width of the Gaussian filter smoothing the structure tensor (local averaging). Defaults to 0 (recommended).
    sampling: Real, list[Real]
            Sampling step (i.e. distance between two consecutive elements of an array).
            Defaults to 1.
    freezing_arr: NDArray
                Array at which the diffusion tensor is evaluated and then frozen.
    matrix_based_impl: bool
                Whether to use matrix based implementation or not. Defaults to False. Recommended to set `True` if
                a ``freezing_arr`` is passed.

    Returns
    -------
    op: OpT
            Anisotropic edge-enhancing diffusion operator.

    Example
    -------

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        import pyxu.opt.solver as pysol
        import pyxu.abc.solver as pysolver
        import pyxu.opt.stop as pystop
        import pyxu_diffops.operator as pyxop
        import skimage as skim

        # Import RGB image
        image = skim.data.cat().astype(float)
        print(image.shape)  # (300, 451, 3)

        # Move color-stacking axis to front (needed for pyxu stacking convention)
        image = np.moveaxis(image, 2, 0)
        print(image.shape)  # (3, 300, 451)

        # Instantiate diffusion operator
        edge_enh_diffop = pyxop.AnisEdgeEnhancingDiffusionOp(dim_shape=(3, 300, 451), beta=10)

        # Define PGD solver, with stopping criterion and starting point x0
        stop_crit = pystop.MaxIter(n=100)

        # Perform 50 gradient flow iterations starting from x0
        PGD = pysol.PGD(f=edge_enh_diffop, show_progress=False, verbosity=100)
        PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=image, stop_crit=stop_crit,
                       tau=2 / edge_enh_diffop.diff_lipschitz))
        edge_enh_image = PGD.solution()

        # Reshape images for plotting.
        image = np.moveaxis(image, 0, 2)
        edge_enh_image = np.moveaxis(edge_enh_image, 0, 2)

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image.astype(int))
        ax[0].set_title("Image", fontsize=15)
        ax[0].axis('off')
        ax[1].imshow(edge_enh_image.astype(int))
        ax[1].set_title("100 iterations Anis. Edge Enhancing", fontsize=15)
        ax[1].axis('off')

    """

    def __init__(
        self,
        dim_shape: pyct.NDArrayShape,
        beta: pyct.Real = 1.0,
        m: pyct.Real = 4.0,
        sigma_gd_st: pyct.Real = 2,
        smooth_sigma_st: pyct.Real = 0,
        sampling: typ.Union[pyct.Real, cabc.Sequence[pyct.Real, ...]] = 1,
        freezing_arr: pyct.NDArray = None,
        matrix_based_impl: bool = False,
        gpu: bool = False,
        dtype: typ.Optional[pyct.DType] = None,
    ):
        gradient = pydiff.Gradient(
            dim_shape=dim_shape,
            directions=(1, 2),
            diff_method="fd",
            scheme="forward",
            mode="symmetric",
            sampling=sampling,
            gpu=gpu,
            dtype=dtype,
        )
        structure_tensor = pyfilt.StructureTensor(
            dim_shape=dim_shape[1:],
            diff_method="gd",
            smooth_sigma=smooth_sigma_st,
            sigma=sigma_gd_st,
            mode="symmetric",
            sampling=sampling,
            gpu=gpu,
            dtype=dtype,
        )
        edge_enh_diffusion_coeff = DiffusionCoeffAnisoEdgeEnhancing(
            dim_shape=dim_shape, structure_tensor=structure_tensor, beta=beta, m=m
        )
        self.freezing_arr = freezing_arr
        if freezing_arr is not None:
            if len(freezing_arr.shape) == len(dim_shape) - 1:
                xp = pycu.get_array_module(freezing_arr)
                self.freezing_arr = xp.expand_dims(freezing_arr, axis=0)
            edge_enh_diffusion_coeff.freeze(self.freezing_arr)
        super().__init__(
            dim_shape,
            gradient=gradient,
            diffusion_coefficient=edge_enh_diffusion_coeff,
            matrix_based_impl=matrix_based_impl,
            gpu=gpu,
            dtype=dtype,
        )
        # initialize lipschitz constants. should this be done instead inside _DiffusionOp.__init__ ??
        self.diff_lipschitz = gradient.lipschitz**2

    # @pycrt.enforce_precision(i="arr")
    # @pycu.vectorize("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.Real:
        if self.freezing_arr is not None:
            xp = pycu.get_array_module(arr)
            y = self._compute_divergence_term(arr)  # (batch,nchannels,nx,ny)
            return xp.einsum("...jkl,...jkl->...", y, arr) / 2  # (batch,)
        else:
            raise RuntimeError(
                "'apply()' method not defined for AnisEdgeEnhancingDiffusionOp if no `freezing_arr` is passed."
            )


class AnisCoherenceEnhancingDiffusionOp(_Diffusion):
    r"""
    Anisotropic coherence-enhancing diffusion operator, featuring an inhomogeneous anisotropic diffusion tensor
    defined as in [Weickert]_. The diffusion tensor is defined as a function of the
    :py:class:`~pyxu.operator.linop.filter.StructureTensor`,
    which is a tensor describing the properties of the image in the neighbourhood of each pixel.
    This diffusion operator allows coherence enhancement by increasing the flux in the direction of the
    eigenvector of the structure tensor with smallest eigenvalue, with an intensity which depends on
    the image coherence, which is measured as :math:`(e_0-e_1)^2`. Stronger coherence leads to stronger smoothing.

    The gradient of the operator reads :math:`-\mathrm{div}(\mathbf{D}(\mathbf{f})\boldsymbol{\nabla}\mathbf{f})`.

    Let :math:`\mathbf{v}_0^i,\mathbf{v}_1^i` be the eigenvectors of the :math:`i`-th pixel structure tensor
    :math:`S_i`, with eigenvalues :math:`e_0^i, e_1^i` sorted in decreasing order. The diffusion tensor is defined,
    for the :math:`i`-th pixel, as the symmetric positive definite tensor:

    .. math::

        \big(\mathbf{D}(\mathbf{f})\big)_i = \lambda_0^i\mathbf{v}_0^i(\mathbf{v}_0^i)^T+\lambda_1^i\mathbf{v}_1^i(\mathbf{v}_1^i)^T,

    where the smoothing intensities along the two eigendirections are defined, to achieve coherence-enhancing, as

    .. math::
        \lambda_0^i &= \alpha,\\
        \lambda_1^i &=  h(e_0^i, e_1^i) :=
        \begin{cases}
              \alpha & \text{if } e_0^i=e_1^i \\
               \alpha + (1-\alpha) \exp \big(\frac{-C}{(e_0^i-e_1^i)^{2m}}\big) & \text{otherwise},
        \end{cases}

    where :math:`\alpha \in (0, 1)` controls the smoothing intensity along the first eigendirection, :math:`m` controls
    the decay rate of :math:`\lambda_0^i` as a function of :math:`(e_0^i-e_1^i)`, and :math:`C\in\mathbb{R}` is a constant.

    For regions with low coherence, smoothing is performed uniformly along all
    directions with intensity :math:`\lambda_1^i \approx \lambda_0^i = \alpha`. For regions with high coherence, we have
    :math:`\lambda_1^i \approx 1 > \lambda_0^i = \alpha`, hence the smoothing intensity is higher in the direction
    orthogonal to the (locally averaged) gradient.

    In general, the diffusion operator does not admit a potential. However, if the diffusion tensor is evaluated
    at some image :math:`\tilde{\mathbf{f}}` and kept fixed to :math:`\mathbf{D}(\tilde{\mathbf{f}})`, the operator
    derives from the potential :math:`\Vert\sqrt{\mathbf{D}(\tilde{\mathbf{f}})}\boldsymbol{\nabla}\mathbf{f}\Vert_2^2`.
    By doing so, the smoothing directions and intensities are fixed according to the features of an image of interest.
    This can be achieved passing a *freezing array* at initialization. In such cases, a *matrix-based implementation*
    is recommended for efficiency.

    Parameters
    ----------
    dim_shape: NDArrayShape
        Shape of the input array.
    alpha: Real
        Anisotropic parameter, determining intensity of smoothing inhibition along the (locally averaged) image gradient.
        Defaults to 1e-1.
    m: Real
        Decay parameter, determines how quickly the smoothing effect changes as a function of :math:`e_0^i/\beta`.
        Defaults to 1.
    sigma_gd_st: Real
       Gaussian width of the gaussian derivative involved in the structure tensor computation
       (if `diff_method_struct_tens="gd"`). Defaults to 2.
    smooth_sigma_st: Real
       Width of the Gaussian filter smoothing the structure tensor (local averaging). Defaults to 4.
    sampling: Real, list[Real]
            Sampling step (i.e. distance between two consecutive elements of an array).
            Defaults to 1.
    diff_method_struct_tens: str
                            Differentiation method for structure tensor computation. Must be either 'gd' or 'fd'.
                            Defaults to 'gd'.
    freezing_arr: NDArray
                Array at which the diffusion tensor is evaluated and then frozen.
    matrix_based_impl: bool
                Whether to use matrix based implementation or not. Defaults to False. Recommended to set `True` if
                a ``freezing_arr`` is passed.

    Returns
    -------
    op: OpT
            Anisotropic coherence-enhancing diffusion operator.

    Example
    -------

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        import pyxu.opt.solver as pysol
        import pyxu.abc.solver as pysolver
        import pyxu.opt.stop as pystop
        import pyxu_diffops.operator as pyxop
        import skimage as skim

        # Import RGB image
        image = skim.data.cat().astype(float)
        print(image.shape)  # (300, 451, 3)

        # Move color-stacking axis to front (needed for pyxu stacking convention)
        image = np.moveaxis(image, 2, 0)
        print(image.shape)  # (3, 300, 451)

        # Instantiate diffusion operator
        coh_enh_diffop = pyxop.AnisCoherenceEnhancingDiffusionOp(dim_shape=(3, 300, 451), alpha=1e-3)

        # Define PGD solver, with stopping criterion and starting point x0
        stop_crit = pystop.MaxIter(n=100)

        # Perform 50 gradient flow iterations starting from x0
        PGD = pysol.PGD(f=coh_enh_diffop, show_progress=False, verbosity=100)
        PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=image, stop_crit=stop_crit,
                       tau=2 / coh_enh_diffop.diff_lipschitz))
        coh_enh_image = PGD.solution()

        # Reshape images for plotting.
        image = np.moveaxis(image, 0, 2)
        coh_enh_image = np.moveaxis(coh_enh_image, 0, 2)

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image.astype(int))
        ax[0].set_title("Image", fontsize=15)
        ax[0].axis('off')
        ax[1].imshow(coh_enh_image.astype(int))
        ax[1].set_title("100 iterations Anis. Coherence Enhancing", fontsize=15)
        ax[1].axis('off')

    """

    def __init__(
        self,
        dim_shape: pyct.NDArrayShape,
        alpha: pyct.Real = 0.1,
        m: pyct.Real = 1,
        sigma_gd_st: pyct.Real = 2,
        smooth_sigma_st: pyct.Real = 4,
        sampling: typ.Union[pyct.Real, cabc.Sequence[pyct.Real, ...]] = 1,
        diff_method_struct_tens: str = "gd",
        freezing_arr: pyct.NDArray = None,
        matrix_based_impl: bool = False,
        gpu: bool = False,
        dtype: typ.Optional[pyct.DType] = None,
    ):
        gradient = pydiff.Gradient(
            dim_shape=dim_shape,
            directions=(1, 2),
            diff_method="fd",
            scheme="forward",
            mode="symmetric",
            sampling=sampling,
            gpu=gpu,
            dtype=dtype,
        )
        structure_tensor = pyfilt.StructureTensor(
            dim_shape=dim_shape[1:],
            diff_method=diff_method_struct_tens,
            smooth_sigma=smooth_sigma_st,
            sigma=sigma_gd_st,
            mode="symmetric",
            sampling=sampling,
            gpu=gpu,
            dtype=dtype,
        )
        coh_enh_diffusion_coeff = DiffusionCoeffAnisoCoherenceEnhancing(
            dim_shape=dim_shape, structure_tensor=structure_tensor, alpha=alpha, m=m
        )
        self.freezing_arr = freezing_arr
        if freezing_arr is not None:
            if len(freezing_arr.shape) == len(dim_shape) - 1:
                xp = pycu.get_array_module(freezing_arr)
                self.freezing_arr = xp.expand_dims(freezing_arr, axis=0)
            coh_enh_diffusion_coeff.freeze(self.freezing_arr)
        super().__init__(
            dim_shape,
            gradient=gradient,
            diffusion_coefficient=coh_enh_diffusion_coeff,
            matrix_based_impl=matrix_based_impl,
            gpu=gpu,
            dtype=dtype,
        )
        # initialize lipschitz constants. should this be done instead inside _DiffusionOp.__init__ ??
        self.diff_lipschitz = gradient.lipschitz**2

    # @pycrt.enforce_precision(i="arr")
    # @pycu.vectorize("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.Real:
        if self.freezing_arr is not None:
            xp = pycu.get_array_module(arr)
            y = self._compute_divergence_term(arr)  # (batch,nchannels,nx,ny)
            return xp.einsum("...jkl,...jkl->...", y, arr) / 2  # (batch,)
        else:
            raise RuntimeError(
                "'apply()' method not defined for AnisCoherenceEnhancingDiffusionOp if no `freezing_arr` is passed."
            )


class AnisDiffusionOp(_Diffusion):
    r"""
    Anisotropic diffusion operator, featuring an inhomogeneous anisotropic diffusion tensor.
    The diffusion tensor is defined as a function of the
    :py:class:`~pyxu.operator.linop.filter.StructureTensor`,
    which is a tensor describing the properties of the image in the neighbourhood of each pixel.
    This diffusion operator allows anisotropic smoothing by reducing the flux in the direction of the
    eigenvector of the structure tensor with largest eigenvalue. Essentially, the (locally averaged)
    gradient is preserved.

    The gradient of the operator reads :math:`-\mathrm{div}(\mathbf{D}(\mathbf{f})\boldsymbol{\nabla}\mathbf{f})`.

    Let :math:`\mathbf{v}_0^i,\mathbf{v}_1^i` be the eigenvectors of the :math:`i`-th pixel structure tensor
    :math:`S_i`, with eigenvalues :math:`e_0^i, e_1^i` sorted in decreasing order. The diffusion tensor is defined,
    for the :math:`i`-th pixel, as the symmetric positive definite tensor:

    .. math::
        \big(\mathbf{D}(\mathbf{f})\big)_i = \lambda_0^i\mathbf{v}_0^i(\mathbf{v}_0^i)^T+\lambda_1^i\mathbf{v}_1^i(\mathbf{v}_1^i)^T,

    where the smoothing intensities along the two eigendirections are defined, to achieve anisotropic smoothing, as

    .. math::
        \lambda_0^i &= \alpha,\\
        \lambda_1^i &=  1,

    where :math:`\alpha \in (0, 1)` controls the smoothing intensity along the first eigendirection.

    In general, the diffusion operator does not admit a potential. However, if the diffusion tensor is evaluated
    at some image :math:`\tilde{\mathbf{f}}` and kept fixed to :math:`\mathbf{D}(\tilde{\mathbf{f}})`, the operator
    derives from the potential :math:`\Vert\sqrt{\mathbf{D}(\tilde{\mathbf{f}})}\boldsymbol{\nabla}\mathbf{f}\Vert_2^2`.
    By doing so, the smoothing directions and intensities are fixed according to the features of an image of interest.
    This can be achieved passing a *freezing array* at initialization. In such cases, a *matrix-based implementation*
    is recommended for efficiency.

    Parameters
    ----------
    dim_shape: NDArrayShape
        Shape of the input array.
    alpha: Real
        Anisotropic parameter, determining intensity of smoothing inhibition along the (locally averaged) image gradient.
        Defaults to 1e-1.
    sigma_gd_st: Real
       Gaussian width of the gaussian derivative involved in the structure tensor computation
       (if `diff_method_struct_tens="gd"`). Defaults to 1.
    smooth_sigma_st: Real
       Width of the Gaussian filter smoothing the structure tensor (local averaging). Defaults to 1.
    sampling: Real, list[Real]
            Sampling step (i.e. distance between two consecutive elements of an array).
            Defaults to 1.
    diff_method_struct_tens: str
                            Differentiation method for structure tensor computation. Must be either 'gd' or 'fd'.
                            Defaults ot 'fd'.
    freezing_arr: NDArray
                Array at which the diffusion tensor is evaluated and then frozen.
    matrix_based_impl: bool
                Whether to use matrix based implementation or not. Defaults to False. Recommended to set `True` if
                a ``freezing_arr`` is passed.

    Returns
    -------
    op: OpT
            Anisotropic diffusion operator.

    """

    def __init__(
        self,
        dim_shape: pyct.NDArrayShape,
        alpha: pyct.Real = 0.1,
        sigma_gd_st: pyct.Real = 1,
        sampling: typ.Union[pyct.Real, cabc.Sequence[pyct.Real, ...]] = 1,
        diff_method_struct_tens: str = "fd",
        freezing_arr: pyct.NDArray = None,
        matrix_based_impl: bool = False,
        gpu: bool = False,
        dtype: typ.Optional[pyct.DType] = None,
    ):
        gradient = pydiff.Gradient(
            dim_shape=dim_shape,
            directions=(1, 2),
            diff_method="fd",
            scheme="forward",
            mode="symmetric",
            sampling=sampling,
            gpu=gpu,
            dtype=dtype,
        )
        structure_tensor = pyfilt.StructureTensor(
            dim_shape=dim_shape[1:],
            diff_method=diff_method_struct_tens,
            smooth_sigma=0,
            sigma=sigma_gd_st,
            mode="symmetric",
            sampling=sampling,
            gpu=gpu,
            dtype=dtype,
        )
        anis_diffusion_coeff = DiffusionCoeffAnisotropic(
            dim_shape=dim_shape, structure_tensor=structure_tensor, alpha=alpha
        )
        self.freezing_arr = freezing_arr
        if freezing_arr is not None:
            if len(freezing_arr.shape) == len(dim_shape) - 1:
                xp = pycu.get_array_module(freezing_arr)
                self.freezing_arr = xp.expand_dims(freezing_arr, axis=0)
            anis_diffusion_coeff.freeze(self.freezing_arr)
        super().__init__(
            dim_shape,
            gradient=gradient,
            diffusion_coefficient=anis_diffusion_coeff,
            matrix_based_impl=matrix_based_impl,
            gpu=gpu,
            dtype=dtype,
        )
        # initialize lipschitz constants. should this be done instead inside _DiffusionOp.__init__ ??
        self.diff_lipschitz = gradient.lipschitz**2

    # @pycrt.enforce_precision(i="arr")
    # @pycu.vectorize("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.Real:
        if self.freezing_arr is not None:
            xp = pycu.get_array_module(arr)
            y = self._compute_divergence_term(arr)  # (batch,nchannels,nx,ny)
            return xp.einsum("...jkl,...jkl->...", y, arr) / 2  # (batch,)
        else:
            raise RuntimeError("'apply()' method not defined for AnisDiffusionOp if no `freezing_arr` is passed.")


# class AnisMfiDiffusionOp(_Diffusion):
#     r"""
#     Example
#     -------
#
#     .. plot::
#
#         import numpy as np
#         import matplotlib.pyplot as plt
#         import pyxu.opt.solver as pysol
#         import pyxu.abc.solver as pysolver
#         import pyxu.opt.stop as pystop
#         import src.diffusion_ops.operator as diffop
#         import skimage as skim
#
#         # Import RGB image
#         image = skim.data.cat().astype(float)
#         print(image.shape) #(300, 451, 3)
#         # move color-stacking axis to front (needed for pyxu stacking convention)
#         image = np.moveaxis(image, 2, 0)
#         print(image.shape) #(3, 300, 451)
#         # Instantiate diffusion operator
#         coh_enh_diffop = diffop.AnisCoherenceEnhancingDiffusionOp(dim_shape=(300, 451), nchannels=3, alpha=1e-3, m=1)
#         # Define PGD solver, with stopping criterion and starting point x0
#         stop_crit = pystop.MaxIter(n=100)
#         x0 = image.reshape(1,-1)
#         # Perform 100 gradient flow iterations starting from x0
#         PGD = pysol.PGD(f = coh_enh_diffop, show_progress=False, verbosity=100)
#         PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=x0, stop_crit=stop_crit,
#                        tau=2/coh_enh_diffop.diff_lipschitz))
#         coh_enhanced_image = PGD.solution()
#         # Reshape images for plotting.
#         image = np.moveaxis(image, 0, 2)
#         coh_enhanced_image = np.moveaxis(coh_enhanced_image.reshape(3, 300, 451), 0, 2)
#         # Plot
#         fig, ax = plt.subplots(1,2,figsize=(10,5))
#         ax[0].imshow(image.astype(int))
#         ax[0].set_title("Image", fontsize=15)
#         ax[0].axis('off')
#         ax[1].imshow(coh_enhanced_image.astype(int))
#         ax[1].set_title("100 iterations Coherence-Enhancing", fontsize=15)
#         ax[1].axis('off')
#
#     """
#
#     def __init__(
#         self,
#         dim_shape: pyct.NDArrayShape,
#         freezing_arr_grad: pyct.NDArray,
#         extra_term: bool = True,
#         beta: pyct.Real = 1,
#         clipping_value: pyct.Real = 1e-5,
#         sigma_gd_st: pyct.Real = 1,
#         alpha: pyct.Real = 0.1,
#         sampling: typ.Union[pyct.Real, cabc.Sequence[pyct.Real, ...]] = 1,
#         diff_method_struct_tens: str = "fd",
#     ):
#         # Needs to be revisited and adjusted to new changes in shapes etc. also, very interesting, because we want to implement
#         # approach shown here by default for all diffusion ops, so we should carefully consider this
#
#         # instantiate custom gradient operator, obtained projecting actual gradient along the structure tensor's eigendirections
#         # define non-oriented gradient matrix of shape (2N, N)
#         Dx = - np.diag(np.ones(dim_shape[0])) + np.diag(np.ones(dim_shape[0] - 1), 1)
#         Dx[-1, -1] = 0  # symmetric boundary conditions, no flux
#         Dy = - np.diag(np.ones(dim_shape[1])) + np.diag(np.ones(dim_shape[1] - 1), 1)
#         Dy[-1, -1] = 0  # symmetric boundary conditions, no flux
#         # define gradient matrix
#         D = np.vstack((np.kron(Dx, np.eye(dim_shape[1])), np.kron(np.eye(dim_shape[0]), Dy)))
#         # instantiate helper reg_fct_matrixfree to access to its methods
#         reg_fct_matrixfree = AnisDiffusionOp(dim_shape=dim_shape,
#                                                   alpha=alpha,
#                                                   sigma_gd_st=sigma_gd_st,
#                                                   sampling=sampling,
#                                                   diff_method_struct_tens=diff_method_struct_tens)
#         reg_fct_matrixfree._op.diffusion_coefficient.alpha = alpha
#         # returns eigenvectors u of shape (N, 2, 2), eigenvalues e of shape (N, 2)
#         u, e = reg_fct_matrixfree._op.diffusion_coefficient._eigendecompose_struct_tensor(freezing_arr_grad.reshape(1, -1))
#         # returns lambdas of shape (N, 2)
#         lambdas = reg_fct_matrixfree._op.diffusion_coefficient._compute_intensities(e)
#         lambdas = np.sqrt(lambdas)
#         # assemble coefficients w_coeffs of shape (2N, 1) to obtain oriented gradient
#         w_coeffs_e1 = (lambdas[:, 0] * u[:, :, 0]).flatten(order='F')
#         w_coeffs_e2 = (lambdas[:, 1] * u[:, :, 1]).flatten(order='F')
#         # compute oriented gradient matrices of shape (2N, N)
#         oriented_gradient_e1 = w_coeffs_e1 * D
#         oriented_gradient_e2 = w_coeffs_e2 * D
#         # assemble oriented gradient matrix by summing the two subblocks
#         N = np.prod(dim_shape)
#         oriented_gradient_matrix = np.vstack((oriented_gradient_e1[:N, :] + oriented_gradient_e1[N:, :],
#                                               oriented_gradient_e2[:N, :] + oriented_gradient_e2[N:, :]))
#         # instantiate gradient operator
#         gradient = pxa.LinOp.from_array(A=sp.csr_matrix(oriented_gradient_matrix))
#
#         # instantiate mfi_diffusion coeff
#         mfi_diffusion_coeff = DiffusionCoeffIsotropic(
#             dim_shape=dim_shape, diffusivity=MfiDiffusivity(dim_shape=dim_shape, beta=beta, clipping_value=clipping_value)
#         )
#         if extra_term:
#             mfi_extra_term = MfiExtraTerm(dim_shape=dim_shape, gradient=gradient, beta=beta, clipping_value=clipping_value)
#             super().__init__(
#                 dim_shape, gradient=gradient, diffusion_coefficient=mfi_diffusion_coeff, extra_diffusion_term=mfi_extra_term
#             )
#         else:
#             super().__init__(
#                 dim_shape, gradient=gradient, diffusion_coefficient=mfi_diffusion_coeff)
#         # initialize lipschitz constants. should this be done instead inside _DiffusionOp.__init__ ??
#         self.diff_lipschitz = gradient.lipschitz ** 2
#         if extra_term:
#             self.diff_lipschitz *= 2
#
#     #@pycrt.enforce_precision(i="arr")
#     @pycu.vectorize("arr")
#     def apply(self, arr: pyct.NDArray) -> pyct.Real:
#         if self.extra_diffusion_term is not None:
#             xp = pycu.get_array_module(arr)
#             y = self._compute_divergence_term(arr)
#             arr = arr.reshape(1, -1)
#             return xp.dot(arr, y.T) / 2
#         else:
#             raise RuntimeError("'apply()' method not defined for AnisMfiDiffusionOp with no MfiExtraTerm: no underlying variational intepretation")
