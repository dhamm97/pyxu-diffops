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
            self.diff_lipschitz = gradient.lipschitz**2 * beta / clipping_value
        # REMARK: I think that with extra term we don't have lipschitzianity, only Holder-continuity order 2.
        # However, I did not observe instability in practice. I think it's due to denominator or order norm(x)!
        # if that's true, a reasonable estimate would be the following (doubling diff_lipschitz)
        if extra_term:
            self.diff_lipschitz *= 2

    # @pycrt.enforce_precision(i="arr")
    # @pycu.vectorize("arr")
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

    # r"""
    # Example
    # -------
    #
    # ..plot::
    #
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     import pyxu.opt.solver as pysol
    #     import pyxu.abc.solver as pysolver
    #     import pyxu.opt.stop as pystop
    #     import pyxu.operator as pyxop
    #     import skimage as skim
    #
    #     # Import RGB image
    #     image = skim.color.rgb2gray(skim.data.cat())
    #     print(image.shape) #(300, 451)
    #     # Instantiate diffusion operator
    #     pm_diffop = pyxop.PeronaMalikDiffusion(dim_shape=(1, 300, 451), beta=0.01)
    #     # Define PGD solver, with stopping criterion and starting point x0
    #     stop_crit = pystop.MaxIter(n=100)
    #     # Perform 50 gradient flow iterations starting from x0
    #     PGD = pysol.PGD(f = pm_diffop, show_progress=False, verbosity=100)
    #     PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=np.expand_dims(image,0), stop_crit=stop_crit,
    #                    tau=2/pm_diffop.diff_lipschitz))
    #     pm_smoothed_image = PGD.solution()
    #     # Plot
    #     fig, ax = plt.subplots(1,2,figsize=(10,5))
    #     ax[0].imshow(image, cmap="gray")
    #     ax[0].set_title("Image", fontsize=15)
    #     ax[0].axis('off')
    #     ax[1].imshow(pm_smoothed_image.squeeze(), cmap="gray")
    #     ax[1].set_title("100 iterations Perona Malik diffusion", fontsize=15)
    #     ax[1].axis('off')
    #     plt.show()
    #
    #     # Import RGB image
    #     image = skim.data.cat().astype(float)
    #     print(image.shape) #(300, 451, 3)
    #     # move color-stacking axis to front (needed for pyxu stacking convention)
    #     image = np.moveaxis(image, 2, 0)
    #     print(image.shape) #(3, 300, 451)
    #     # Instantiate diffusion operator
    #     pm_diffop = pyxop.PeronaMalikDiffusion(dim_shape=(3, 300, 451), beta=5)
    #     # Define PGD solver, with stopping criterion and starting point x0
    #     stop_crit = pystop.MaxIter(n=100)
    #     # Perform 50 gradient flow iterations starting from x0
    #     PGD = pysol.PGD(f = pm_diffop, show_progress=False, verbosity=100)
    #     PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=image, stop_crit=stop_crit,
    #                    tau=2/pm_diffop.diff_lipschitz))
    #     pm_smoothed_image = PGD.solution()
    #     # Reshape images for plotting.
    #     image = np.moveaxis(image, 0, 2)
    #     pm_smoothed_image = np.moveaxis(pm_smoothed_image, 0, 2)
    #     # Plot
    #     fig, ax = plt.subplots(1,2,figsize=(10,5))
    #     ax[0].imshow(image.astype(int))
    #     ax[0].set_title("Image", fontsize=15)
    #     ax[0].axis('off')
    #     ax[1].imshow(pm_smoothed_image.astype(int))
    #     ax[1].set_title("100 iterations Perona Malik diffusion", fontsize=15)
    #     ax[1].axis('off')
    #     plt.show()
    #
    # """
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

    # @pycrt.enforce_precision(i="arr")
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

    # @pycrt.enforce_precision(i="arr")
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

    # @pycrt.enforce_precision(i="arr")
    # @pycu.vectorize("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        f = dict(
            exponential=self._apply_exponential,
            rational=self._apply_rational,
        ).get(self.pm_fct)
        out = f(arr)
        return out


class TikhonovDiffusion(_Diffusion):
    r"""
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

    # @pycrt.enforce_precision(i="arr")
    # @pycu.vectorize("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.Real:
        xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
        y = self._compute_divergence_term(arr)  # (batch,nchannels,nx,ny)
        return xp.einsum("...jkl,...jkl->...", y, arr) / 2  # (batch,)


class TotalVariationDiffusion(_Diffusion):
    r"""
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

    # r"""
    # Example
    # -------
    #
    # .. plot::
    #
    #     import matplotlib.pyplot as plt
    #     import pyxu.opt.solver as pysol
    #     import pyxu.abc.solver as pysolver
    #     import pyxu.opt.stop as pystop
    #     import pyxu.operator as pyxop
    #     import skimage as skim
    #
    #     # Import RGB image
    #     image = skim.color.rgb2gray(skim.data.cat())
    #     print(image.shape) #(300, 451)
    #     # Instantiate diffusion operator
    #     tv_diffop = pyxop.TotalVariationDiffusion(dim_shape=(1, 300, 451), beta=0.005)
    #     # Define PGD solver, with stopping criterion and starting point x0
    #     stop_crit = pystop.MaxIter(n=100)
    #     # Perform 50 gradient flow iterations starting from x0
    #     PGD = pysol.PGD(f = tv_diffop, show_progress=False, verbosity=100)
    #     PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=np.expand_dims(image,0), stop_crit=stop_crit,
    #                    tau=2/tv_diffop.diff_lipschitz))
    #     tv_smoothed_image = PGD.solution()
    #     # Plot
    #     fig, ax = plt.subplots(1,2,figsize=(10,5))
    #     ax[0].imshow(image, cmap="gray")
    #     ax[0].set_title("Image", fontsize=15)
    #     ax[0].axis('off')
    #     ax[1].imshow(tv_smoothed_image.squeeze(), cmap="gray")
    #     ax[1].set_title("100 iterations Total Variation smoothing", fontsize=15)
    #     ax[1].axis('off')
    #     plt.show()
    #
    # """

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
    Class for curvature preserving diffusion operators [see `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_].

    This class provides an interface to deal with curvature preserving diffusion operators in the context of PDE-based regularisation.
    In particular, we consider diffusion processes that can be written as

    .. math::
        \frac{\partial\mathbf{f}}{\partial t} = \mathbf{T}_{out}\mathrm{trace}\big(\mathbf{T}_{in}\mathbf{H}(\mathbf{f})\big) + (\nabla\mathbf{f})^T \mathbf{J}_{\mathbf{w}}\mathbf{w},

    where
        * :math:`\mathbf{T}_{out} = \mathbf{T}_{out}(\mathbf{f})` is the outer diffusivity for the trace term;
        * :math:`\mathbf{T}_{in} = \mathbf{T}_{in}(\mathbf{f})` is the diffusion coefficient for the trace term;
        * :math:`\mathbf{w}` is a vector field assigning a :math:`2`-dimensional vector to each pixel;
        * :math:`\mathbf{J}_\mathbf{w}` is the Jacobian of the vector field :math:`\mathbf{w}`.

    The right-hand side of the above PDE represents the output of the ``grad()`` method applied to the image :math:`\mathbf{f}`.

    The resulting smoothing process tries to preserve the curvature of the vector field :math:`\mathbf{w}`.

    The effect of the :py:class:`~pyxu.operator.diffusion.CurvaturePreservingDiffusionOp` on an image :math:`\mathbf{f}` can be better understood
    by focusing on a single pixel :math:`f_i` of the vectorisation of :math:`\mathbf{f}` (see, e.g., discussion in
    :py:class:`~pyxu.operator.diffusion._DiffusionCoefficient`).

    Example
    -------

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        import pyxu.operator.linop.diff as pydiff
        import pyxu.opt.solver as pysol
        import pyxu.abc.solver as pysolver
        import pyxu.opt.stop as pystop
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
        fig, ax = plt.subplots(1, 3, figsize=(20, 4))
        ax[0].imshow(image.astype(int), cmap="gray", aspect="auto")
        ax[0].set_title("Image")
        ax[0].axis('off')
        ax[1].quiver(curv_pres_2[::40, ::60], curv_pres_1[::40, ::60])
        ax[1].set_title("Vector field")
        ax[1].axis('off')
        ax[2].imshow(curv_smooth_image.astype(int), cmap="gray", aspect="auto")
        ax[2].set_title("200 iterations Curvature Preserving")
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
    Example
    -------

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        import pyxu.opt.solver as pysol
        import pyxu.abc.solver as pysolver
        import pyxu.opt.stop as pystop
        import src.diffusion_ops.operator as diffop
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
    ):
        gradient = pydiff.Gradient(
            dim_shape=dim_shape,
            directions=(1, 2),
            diff_method="fd",
            scheme="forward",
            mode="symmetric",
            sampling=sampling,
        )
        structure_tensor = pyfilt.StructureTensor(
            dim_shape=dim_shape[1:],
            diff_method="gd",
            smooth_sigma=smooth_sigma_st,
            sigma=sigma_gd_st,
            mode="symmetric",
            sampling=sampling,
        )
        edge_enh_diffusion_coeff = DiffusionCoeffAnisoEdgeEnhancing(
            dim_shape=dim_shape, structure_tensor=structure_tensor, beta=beta, m=m
        )
        self.freezing_arr = freezing_arr
        if self.freezing_arr is not None:
            edge_enh_diffusion_coeff.freeze(self.freezing_arr)
        super().__init__(dim_shape, gradient=gradient, diffusion_coefficient=edge_enh_diffusion_coeff)
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
    Example
    -------

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        import pyxu.opt.solver as pysol
        import pyxu.abc.solver as pysolver
        import pyxu.opt.stop as pystop
        import src.diffusion_ops.operator as diffop
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
    ):
        gradient = pydiff.Gradient(
            dim_shape=dim_shape,
            directions=(1, 2),
            diff_method="fd",
            scheme="forward",
            mode="symmetric",
            sampling=sampling,
        )
        structure_tensor = pyfilt.StructureTensor(
            dim_shape=dim_shape[1:],
            diff_method=diff_method_struct_tens,
            smooth_sigma=smooth_sigma_st,
            sigma=sigma_gd_st,
            mode="symmetric",
            sampling=sampling,
        )
        coh_enh_diffusion_coeff = DiffusionCoeffAnisoCoherenceEnhancing(
            dim_shape=dim_shape, structure_tensor=structure_tensor, alpha=alpha, m=m
        )
        self.freezing_arr = freezing_arr
        if self.freezing_arr is not None:
            coh_enh_diffusion_coeff.freeze(freezing_arr)
        super().__init__(dim_shape, gradient=gradient, diffusion_coefficient=coh_enh_diffusion_coeff)
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
    Example
    -------

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        import pyxu.opt.solver as pysol
        import pyxu.abc.solver as pysolver
        import pyxu.opt.stop as pystop
        import src.diffusion_ops.operator as diffop
        import skimage as skim

        # Import RGB image
        image = skim.data.cat().astype(float)
        print(image.shape)  # (300, 451, 3)

        # Move color-stacking axis to front (needed for pyxu stacking convention)
        image = np.moveaxis(image, 2, 0)
        print(image.shape)  # (3, 300, 451)

        # Instantiate diffusion operator
        anis_diffop = pyxop.AnisDiffusionOp(dim_shape=(3, 300, 451), alpha=1e-3)

        # Define PGD solver, with stopping criterion and starting point x0
        stop_crit = pystop.MaxIter(n=100)

        # Perform 50 gradient flow iterations starting from x0
        PGD = pysol.PGD(f=anis_diffop, show_progress=False, verbosity=100)
        PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=image, stop_crit=stop_crit,
                       tau=2 / anis_diffop.diff_lipschitz))
        anis_image = PGD.solution()

        # Reshape images for plotting.
        image = np.moveaxis(image, 0, 2)
        anis_image = np.moveaxis(anis_image, 0, 2)

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image.astype(int))
        ax[0].set_title("Image", fontsize=15)
        ax[0].axis('off')
        ax[1].imshow(anis_image.astype(int))
        ax[1].set_title("100 iterations Anisotropic smoothing", fontsize=15)
        ax[1].axis('off')

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
    ):
        gradient = pydiff.Gradient(
            dim_shape=dim_shape,
            directions=(1, 2),
            diff_method="fd",
            scheme="forward",
            mode="symmetric",
            sampling=sampling,
        )
        structure_tensor = pyfilt.StructureTensor(
            dim_shape=dim_shape[1:],
            diff_method=diff_method_struct_tens,
            smooth_sigma=0,
            sigma=sigma_gd_st,
            mode="symmetric",
            sampling=sampling,
        )
        anis_diffusion_coeff = DiffusionCoeffAnisotropic(
            dim_shape=dim_shape, structure_tensor=structure_tensor, alpha=alpha
        )
        self.freezing_arr = freezing_arr
        if self.freezing_arr is not None:
            anis_diffusion_coeff.freeze(freezing_arr)
        super().__init__(
            dim_shape,
            gradient=gradient,
            diffusion_coefficient=anis_diffusion_coeff,
            matrix_based_impl=matrix_based_impl,
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
