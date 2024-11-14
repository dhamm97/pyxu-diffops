import warnings

import numpy as np
import pyxu.abc as pyca
import pyxu.info.ptype as pyct
import pyxu.util as pycu

__all__ = [
    "_Diffusivity",
    "TikhonovDiffusivity",
    "MfiDiffusivity",
    "PeronaMalikDiffusivity",
    "TotalVariationDiffusivity",
]


class _Diffusivity(pyca.Map):
    r"""
    Abstract diffusivity operator. Daughter classes implement specific diffusivity functions.

    Notes
    -----
    Given a :math:`D`-dimensional signal,

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0} \times \cdots \times N_{D-1}},

    its ``apply()`` method returns the :math:`D`-dimensional signal

    .. math::

        g(\mathbf{f}) \in \mathbb{R}^{N_{0} \times \cdots \times N_{D-1}},

    where :math:`g(\cdot)` is known as the *diffusivity function*.

    **Remark 1**

    The class features a ``freeze()`` method. When applied to an array `arr`, it freezes the diffusivity
    at the value obtained applying ``apply()``  to `arr`.

    The class also features a ``set_frozen_diffusivity()`` method. When applied to an array `arr`, it freezes the diffusivity
    at the value `arr`.

    **Remark 2**

    The class features a ``energy_functional()`` method, which can be used to evaluate the energy potential
    that a divergence-based diffusion term featuring the considered diffusivity derives from (when it makes sense).
    When implementing a new diffusivity, one should check whether this variational interpretation holds: if this is the case,
    attribute ``from_potential`` should be set to ``True`` and the method ``energy_functional()`` should be implemented.

    **Remark 3**

    The class features the attribute ``bounded``. If ``True``, this signals that the map returns values
    in the range :math:`(0, 1]`. When implementing a new diffusivity, one should check whether this holds: if this is the case,
    ``bounded`` should be set to ``True``.
    """

    def __init__(self, dim_shape: pyct.NDArrayShape, gradient: pyct.OpT = None):
        """
        Parameters
        ----------
        dim_shape: tuple
            Shape of the input array.
        gradient: :py:class:`~pyxu.operator.linop.diff.Gradient`
            Gradient operator. Defaults to `None`.
        """
        self.nchannels = dim_shape[0]
        super().__init__(dim_shape=dim_shape, codim_shape=dim_shape)
        self.gradient = gradient
        if gradient is not None:
            msg = "`gradient.dim_shape`={} inconsistent with `dim_shape`={}.".format(gradient.dim_shape, dim_shape)
            assert gradient.dim_shape == dim_shape, msg
        self.frozen = False
        self.frozen_diffusivity = None
        self.upper_bound = np.inf

    def freeze(self, arr: pyct.NDArray):
        if self.frozen:
            warnings.warn("Diffusivity has already been frozen. Cannot overwrite previous frozen state.")
        else:
            self.frozen_diffusivity = self.apply(arr)
            self.frozen = True

    def set_frozen_diffusivity(self, arr: pyct.NDArray):
        if self.frozen:
            warnings.warn("Diffusivity has already been frozen. Cannot overwrite previous frozen state.")
        else:
            self.frozen_diffusivity = arr
            self.frozen = True

    # @pycrt.enforce_precision(i="arr")
    def _compute_grad_norm_sq(self, arr: pyct.NDArray, gradient: pyct.OpT = None) -> pyct.NDArray:
        r"""

        Notes
        -------
        If ``arr.shape[0]`` is not `1`, the input is considered multichannel and the Di Zenzo norm is used
        [see `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_]. This corresponds to summing
        up the information from the three channels to have a single "global" diffusivity. Essentially, the
        Di Zenzo gradient norm represents the norm across channels.

        """
        # compute squared norm of gradient (on each pixel), needed for several diffusivities.
        xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
        if gradient is None:
            z = self.gradient(arr)  # (batch,ndims,nchannels,nx,ny)
        else:
            z = gradient(arr)  # (batch,ndims,nchannels,nx,ny)
        z **= 2
        # out = xp.sum(z, axis=(0, 1), keepdims=False)
        # return xp.expand_dims(out, axis=0)
        # out = xp.sum(z, axis=(1, 2), keepdims=False)
        # return out
        # NEW
        z = xp.sum(z, axis=(-4, -3), keepdims=False)  # True  # (batch,nx,ny) if False    (batch,1,1,nx,ny) if True
        # return z
        out = xp.stack([z] * self.nchannels, axis=-3)  # (batch,nchannels,nx,ny)
        return out

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        raise NotImplementedError


class TikhonovDiffusivity(_Diffusivity):
    r"""
    Diffusivity associated to Tikhonov regularization. Leads to isotropic Laplacian diffusion in the context of
    diffusion processes.

    Let :math:`f_i` be an entry (pixel) of the vectorisation of the :math:`D`-dimensional signal,

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0} \times \cdots \times N_{D-1}}.

    Then the Tikhonov diffusivity function reads

    .. math ::

        (g(\mathbf{f}))_i = 1, \quad \forall i.

    Example
    -------

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        #import pyxu.operator.diffusion as pydiffusion
        import pyxu_diffops.operator.diffusion._diffusivity as pydiffusion
        import skimage as skim
        # Import image
        image = skim.color.rgb2gray(skim.data.cat())
        # Define Tikhonov diffusivity
        Tikhonov_diffusivity = pydiffusion.TikhonovDiffusivity(dim_shape=(1, *image.shape))
        # Evaluate diffusivity at image
        Tikhodiff_eval = Tikhonov_diffusivity(np.expand_dims(image, 0))
        # Plot
        fig, ax = plt.subplots(1,3,figsize=(25,5))
        p0=ax[0].imshow(image, cmap="gray", aspect="auto")
        ax[0].set_title("Image", fontsize=15, pad=10)
        ax[0].set_title("Image", fontsize=15, pad=10)
        ax[0].axis('off')
        plt.colorbar(p0, ax=ax[0], fraction=0.04, pad=0.01)
        x=np.linspace(0,1,100)
        ax[1].plot(x, np.ones(x.size))
        ax[1].set_xlabel(r'$f$', fontsize=15)
        ax[1].set_ylabel(r'$g$', fontsize=15, rotation=0, labelpad=7)
        ax[1].set_xlim([0,1])
        ax[1].set_title("Tikhonov diffusivity function", fontsize=15, pad=10)
        p2=ax[2].imshow(Tikhodiff_eval.squeeze(), cmap="gray", aspect="auto")
        ax[2].set_title("Tikhonov diffusivity evaluated at image", fontsize=15, pad=10)
        ax[2].axis('off')
        plt.colorbar(p2, ax=ax[2], fraction=0.04)
        plt.show()

    """

    def __init__(self, dim_shape: pyct.NDArrayShape):
        """

        Parameters
        ----------
        dim_shape: tuple
            Shape of the input array.
        """
        super().__init__(dim_shape=dim_shape)
        self.upper_bound = 1

    # @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        if not self.frozen:
            xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
            y = xp.ones(arr.shape)
            self.set_frozen_diffusivity(y)
            return y  # (batch,nchannels,nx,ny)
        else:
            return self.frozen_diffusivity


class MfiDiffusivity(_Diffusivity):
    r"""
    Minimum Fisher Information (MFI) inspired diffusivity [see `Anton <https://iopscience.iop.org/article/10.1088/0741-3335/38/11/001/pdf>`_].

    Let :math:`f_i` be an entry (pixel) of the vectorisation of the :math:`D`-dimensional signal,

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0} \times \cdots \times N_{D-1}}.

    Then the MFI diffusivity function reads

    * If ``tame=False``:
    .. math ::

        (g(\mathbf{f}))_i = \frac{1} { \max \{ 0, f_i \} / \beta}, \quad \forall i;

    * If ``tame=True``:
    .. math ::

        (g(\mathbf{f}))_i = \frac{1} {1 + \max \{ 0, f_i \} / \beta}, \quad \forall i.


    **Remark 1**

    In both cases, the corresponding divergence-based diffusion term does not allow a variational interpretation (see
    :py:class:`~pyxu.operator.diffusion._DiffusionOp` documentation). Indeed, the Euler-Lagrange equations arising
    from the original variational formulation yield an extra term that cannot be written in divergence form.

    **Remark 2**

    It is recommended to set ``tame=True`` to avoid unstable behavior when the diffusivity is used in the context
    of diffusion processes.

    Example
    -------

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        import pyxu.operator.linop.diff as pydiff
        #import pyxu.operator.diffusion as pydiffusion
        import pyxu_diffops.operator.diffusion._diffusivity as pydiffusion
        import skimage as skim
        # Import image
        image = skim.color.rgb2gray(skim.data.cat())
        # Instantiate gaussian gradient operator
        # Define MFI diffusivity
        MFI_diffusivity = pydiffusion.MfiDiffusivity(dim_shape=(1,*image.shape), tame=True)
        # Evaluate diffusivity at image
        MFIdiff_eval = MFI_diffusivity(np.expand_dims(image, 0))
        # Plot
        fig, ax = plt.subplots(1,3,figsize=(25,5))
        p0=ax[0].imshow(image, cmap="gray", aspect="auto")
        ax[0].set_title("Image", fontsize=15, pad=10)
        ax[0].axis('off')
        plt.colorbar(p0, ax=ax[0], fraction=0.04, pad=0.01)
        x=np.linspace(0,1,100)
        ax[1].plot(x, 1/(1+x))
        ax[1].set_xlabel(r'$f$', fontsize=15)
        ax[1].set_ylabel(r'$g$', fontsize=15, rotation=0, labelpad=7)
        ax[1].set_xlim([0,1])
        ax[1].set_title("Tame MFI diffusivity function", fontsize=15, pad=10)
        p2=ax[2].imshow(MFIdiff_eval.squeeze(), cmap="gray", aspect="auto")
        ax[2].set_title("MFI diffusivity evaluated at image", fontsize=15, pad=10)
        ax[2].axis('off')
        plt.colorbar(p2, ax=ax[2], fraction=0.04)
        plt.show()

    """

    def __init__(
        self, dim_shape: pyct.NDArrayShape, beta: pyct.Real = 1.0, clipping_value: pyct.Real = 1e-5, tame: bool = True
    ):
        """

        Parameters
        ----------
        dim_shape: tuple
            Shape of the input array.
        beta: pyct.Real
            Contrast parameter. Defaults to `1`.
        clipping_value: pyct.Real
            Contrast parameter. Defaults to `1e-5`.
        tame: bool
            Whether to consider the tame version bounded in :math:`(0, 1]` or not. Defaults to `True`.
        """
        super().__init__(dim_shape=dim_shape)
        assert beta > 0, "`beta` must be strictly positive"
        self.beta = beta
        self.clipping_value = clipping_value
        self.tame = tame
        if tame:
            self.upper_bound = 1
        else:
            self.upper_bound = 1 / clipping_value

    # @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        if not self.frozen:
            xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
            # clipped below 1e-5 for consistency with SPC implementation
            # in SPC code, also normalization by mean(arr), somewhat the role of our beta?
            # arr /= xp.mean(arr)
            # y = arr / xp.mean(arr)
            # y = xp.clip(arr, 1e-5, None)

            # y = arr / xp.mean(arr) this was uncommented!

            y = xp.clip(arr, self.clipping_value, None)
            if self.tame:
                y /= self.beta
                y += 1
            return 1 / y  # (batch,nchannels,nx,ny)
        else:
            return self.frozen_diffusivity


class PeronaMalikDiffusivity(_Diffusivity):
    r"""
    Perona-Malik diffusivity [see `Perona-Malik <http://image.diku.dk/imagecanon/material/PeronaMalik1990.pdf>`_].

    Let :math:`f_i` be the :math:`i`-th entry (pixel) of the vectorisation of the :math:`D`-dimensional signal,

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0} \times \cdots \times N_{D-1}}.

    Furthermore, let :math:`(\nabla \mathbf{f})_i = \big((\frac{\partial \mathbf{f}}{\partial x_0})_i,\dots,(\frac{\partial \mathbf{f}}{\partial x_{D-1}})_i\big)^T`.

    Then the Perona-Malik diffusivity function reads

    * In the exponential case:
    .. math ::

        (g(\mathbf{f}))_i = \exp(-\vert (\nabla \mathbf{f})_i \vert ^2 / \beta^2), \quad \forall i;

    * In the rational case:
    .. math ::

        (g(\mathbf{f}))_i = \frac{1} { 1+\vert (\nabla \mathbf{f})_i \vert ^2 / \beta^2}, \quad \forall i,

    where :math:`\beta` is the contrast parameter.

    In both cases, the corresponding divergence-based diffusion term allows a variational interpretation
    [see `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_ for exponential case].

    **Remark**

    It is recommended to provide a Gaussian-derivative-based gradient (:math:`\nabla=\nabla_\sigma`). This acts as regularization
    when the diffusivity is used  for the ill-posed Perona-Malik diffusion process [see `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_].

    Example
    -------

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        import pyxu.operator.linop.diff as pydiff
        #import pyxu.operator.diffusion as pydiffusion
        import pyxu_diffops.operator.diffusion._diffusivity as pydiffusion
        import skimage as skim
        # Import image
        image = skim.color.rgb2gray(skim.data.cat())
        # Instantiate gaussian gradient operator
        gauss_grad = pydiff.Gradient(dim_shape=(1,*image.shape), directions=(1,2), diff_method="gd", sigma=2)
        # Set contrast parameter beta (heuristic)
        gauss_grad_norm = np.linalg.norm(gauss_grad(np.expand_dims(image,0)), axis=0)
        beta = np.quantile(gauss_grad_norm.flatten(), 0.9)
        # Define Perona-Malik diffusivity
        PeronaMalik_diffusivity = pydiffusion.PeronaMalikDiffusivity(dim_shape=(1,*image.shape), gradient=gauss_grad, beta=beta, pm_fct="exponential")
        # Evaluate diffusivity at image
        PMdiff_eval = PeronaMalik_diffusivity(np.expand_dims(image, 0))
        # Plot
        fig, ax = plt.subplots(1,3,figsize=(25,5))
        ax[0].imshow(image, cmap="gray", aspect="auto")
        ax[0].set_title("Image", fontsize=15, pad=10)
        ax[0].axis('off')
        x=np.linspace(0,0.25,100)
        ax[1].plot(x, np.exp(-x**2/(beta**2)))
        ax[1].set_xlabel(r'$\vert \nabla_\sigma f \vert$', fontsize=15)
        ax[1].set_ylabel(r'$g$', fontsize=15, rotation=0, labelpad=10)
        ax[1].set_xlim([0,0.25])
        ax[1].set_title("Exponential Perona-Malik diffusivity function", fontsize=15, pad=10)
        p=ax[2].imshow(PMdiff_eval.squeeze(), cmap="gray", aspect="auto")
        ax[2].set_title("Perona-Malik diffusivity evaluated at image", fontsize=15, pad=10)
        ax[2].axis('off')
        plt.colorbar(p, ax=ax[2], fraction=0.04)
        plt.show()

    """

    def __init__(
        self, dim_shape: pyct.NDArrayShape, gradient: pyct.OpT, beta: pyct.Real = 1, pm_fct: str = "exponential"
    ):
        """

        Parameters
        ----------
        dim_shape: tuple
            Shape of the input array.
        gradient: :py:class:`~pyxu.operator.linop.diff.Gradient`
            Gradient operator. Defaults to `None`.
        beta: pyct.Real
            Contrast parameter. Defaults to `1`.
        pm_fct: str
            Perona-Malik function type. Defaults to `exponential`. Allowed values are `exponential`, `rational`.
        """
        assert pm_fct in ["exponential", "rational"], "Unknown `pm_fct`, allowed values are `exponential`, `rational`."
        super().__init__(dim_shape=dim_shape, gradient=gradient)
        self.beta = beta
        self.pm_fct = pm_fct
        self.upper_bound = 1

    # @pycrt.enforce_precision(i="arr")
    def _apply_exponential(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        # Inplace implementation of
        #   xp.exp(-grad_norm_sq/beta**2)
        y = self._compute_grad_norm_sq(arr)  # (batch,nchannels,nx,ny)
        y /= self.beta**2
        return xp.exp(-y)  # (batch,nchannels,nx,ny)

    # @pycrt.enforce_precision(i="arr")
    def _apply_rational(self, arr: pyct.NDArray) -> pyct.NDArray:
        # Inplace implementation of
        #   1 / (1 + grad_norm_sq/beta**2)
        y = self._compute_grad_norm_sq(arr)
        y /= self.beta**2
        y += 1
        return 1 / y  # (batch,nchannels,nx,ny)

    # @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        f = dict(
            exponential=self._apply_exponential,
            rational=self._apply_rational,
        ).get(self.pm_fct)
        if not self.frozen:
            out = f(arr)
            return out
        else:
            return self.frozen_diffusivity


class TotalVariationDiffusivity(_Diffusivity):
    r"""
    Total Variation (TV) diffusivity [see `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_].

    Let :math:`f_i` be an entry (pixel) of the vectorisation of the :math:`D`-dimensional signal,

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0} \times \cdots \times N_{D-1}}.

    Furthermore, let :math:`(\nabla \mathbf{f})_i = \big((\frac{\partial \mathbf{f}}{\partial x_0})_i,\dots,(\frac{\partial \mathbf{f}}{\partial x_{D-1}})_i\big)^T`.

    Then the Total Variation diffusivity function reads

    * If ``tame=False``:
    .. math ::

        (g(\mathbf{f}))_i = \frac{1} { \vert (\nabla \mathbf{f})_i \vert}, \quad \forall i;

    * If ``tame=True``:
    .. math ::

        (g(\mathbf{f}))_i = \frac{\beta} { \sqrt{\beta^2+ \vert (\nabla \mathbf{f})_i \vert ^2}}, \quad \forall i,

    where :math:`\beta` controls the quality of the smooth approximation of the :math:`L^2`-norm involved in the TV
    approach. The `tame` formulation amounts to an approximation very similar to the Huber loss approach. Lower values
    correspond to better approximations but typically lead to larger computational cost.

    In both cases, the corresponding divergence-based diffusion term allows a variational interpretation
    [see `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_ for untamed case].

    **Remark 1**

    It is recommended to set ``tame=True`` to avoid unstable behavior when the diffusivity is used in the context
    of diffusion processes.

    Example
    -------

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        import pyxu.operator.linop.diff as pydiff
        #import pyxu.operator.diffusion as pydiffusion
        import pyxu_diffops.operator.diffusion._diffusivity as pydiffusion
        import skimage as skim
        # Import image
        image = skim.color.rgb2gray(skim.data.cat())
        # Instantiate gaussian gradient operator
        grad = pydiff.Gradient(
            dim_shape=(1,*image.shape), directions=(1,2), diff_method="fd", scheme="forward",
            mode="symmetric")
        # Set contrast parameter beta (heuristic)
        grad_norm = np.linalg.norm(grad(np.expand_dims(image,0)), axis=0)
        beta = np.quantile(grad_norm.flatten(), 0.9)
        # Define Total Variation diffusivity
        TotalVariation_diffusivity = pydiffusion.TotalVariationDiffusivity(dim_shape=(1,*image.shape), gradient=grad, beta=beta, tame=True)
        # Evaluate diffusivity at image
        TVdiff_eval = TotalVariation_diffusivity(np.expand_dims(image, 0))
        # Plot
        fig, ax = plt.subplots(1,3,figsize=(25,5))
        ax[0].imshow(image, cmap="gray", aspect="auto")
        ax[0].set_title("Image", fontsize=15, pad=10)
        ax[0].axis('off')
        x=np.linspace(0,0.25,100)
        ax[1].plot(x, 1/np.sqrt(1+x**2/(beta**2)))
        ax[1].set_xlabel(r'$\vert \nabla_\sigma f \vert$', fontsize=15)
        ax[1].set_ylabel(r'$g$', fontsize=15, rotation=0, labelpad=10)
        ax[1].set_xlim([0,0.25])
        ax[1].set_title("Total Variation diffusivity", fontsize=15, pad=10)
        p=ax[2].imshow(TVdiff_eval.squeeze(), cmap="gray", aspect="auto")
        ax[2].set_title("Total Variation diffusivity evaluated at image", fontsize=15, pad=10)
        ax[2].axis('off')
        plt.colorbar(p, ax=ax[2], fraction=0.04)
        plt.show()

    """

    def __init__(self, dim_shape: pyct.NDArrayShape, gradient: pyct.OpT, beta: pyct.Real = 1e-3, tame: bool = True):
        """

        Parameters
        ----------
        dim_shape: tuple
            Shape of the input array.
        gradient: :py:class:`~pyxu.operator.linop.diff.Gradient`
            Gradient operator. Defaults to `None`.
        beta: pyct.Real
            Quality of :math:`L^2`-norm smooth approximation. Defaults to :math:`10^{-3}`. Lower values yield better
            approximations, but typically larger computational cost.
        tame: bool
            Whether to consider tame version or not. Defaults to `True`. Non-tame version will likely lead to instability.
        """
        super().__init__(dim_shape=dim_shape, gradient=gradient)
        self.tame = tame
        self.beta = beta
        if tame:
            self.upper_bound = 1

    # @pycrt.enforce_precision(i="arr")
    def _apply_tame(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        # Inplace implementation of
        #   1/(xp.sqrt(1+grad_norm_sq/beta**2) = beta/(xp.sqrt(beta**2+grad_norm_sq)
        y = self._compute_grad_norm_sq(arr)
        y /= self.beta**2
        y += 1
        y = xp.sqrt(y)
        return 1 / y

    # @pycrt.enforce_precision(i="arr")
    def _apply_untamed(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = self._compute_grad_norm_sq(arr)
        return 1 / y

    # @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        f = self._apply_tame if self.tame else self._apply_untamed
        if not self.frozen:
            out = f(arr)
            return out
        else:
            return self.frozen_diffusivity
