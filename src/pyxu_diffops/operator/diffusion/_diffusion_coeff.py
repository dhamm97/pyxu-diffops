import functools
import warnings

import numpy as np
import pyxu.abc as pyca
import pyxu.info.deps as pycd
import pyxu.info.ptype as pyct
import pyxu.util as pycu
import scipy.optimize as sciop

from pyxu_diffops.operator.diffusion._diffusivity import TikhonovDiffusivity

__all__ = [
    "_DiffusionCoefficient",
    "DiffusionCoeffIsotropic",
    "_DiffusionCoeffAnisotropic",
    "DiffusionCoeffAnisoEdgeEnhancing",
    "DiffusionCoeffAnisoCoherenceEnhancing",
    "DiffusionCoeffAnisotropic",
]


class _DiffusionCoefficient(pyca.Map):
    r"""
    Abstract class for (tensorial) diffusion coefficients. Daughter classes :py:class:`~pyxu.operator.diffusion.DiffusionCoeffIsotropic`
    and :py:class:`~pyxu.operator.diffusion._DiffusionCoeffAnisotropic` handle the isotropic/anisotropic cases.

    Notes
    -----
    Given a :math:`D`-dimensional signal:

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0} \times \cdots \times N_{D-1}},

    ``_DiffusionCoefficient`` operators :math:`\mathbf{D}` can be used to define diffusion operators

    * within a divergence-based term
    .. math::
        \mathrm{div} (\mathbf{D} \nabla \mathbf{f}), \qquad \text{where } \mathbf{D} \in
        \mathbb{R}^{D N_{tot} \times D N_{tot} }

    * within a trace-based term
    .. math::
        \mathrm{trace}(\mathbf{D} \mathbf{H}(\mathbf{f})), \qquad \text{where } \mathbf{D} \in
        \mathbb{R}^{D^2 N_{tot} \times D^2 N_{tot} }
    where :math:`\mathbf{H}(\cdot)` is the Hessian.


    **Remark 1**

    In principle ``_DiffusionCoefficient`` depends on the input signal itself (or on some other quantity), so
    that :math:`\mathbf{D}=\mathbf{D}(\mathbf{f})`. The ``apply()`` method, when applied to an array `arr`, returns
    the operator associated to the diffusion coefficient evaluated at `arr`.

    **Remark 2**

    The effect of the ``_DiffusionCoefficient`` :math:`\mathbf{D}` can be better understood by focusing on the :math:`i`-th entry (pixel) :math:`f_i`
    of the vectorisation of :math:`\mathbf{f}`. Furthermore, let
    :math:`(\nabla \mathbf{f})_i = \big((\frac{\partial \mathbf{f}}{\partial x_0})_i,\dots,(\frac{\partial \mathbf{f}}{\partial x_{D-1}})_i\big)^T`.
    We consider the divergence-based case. Then, the :math:`i`-th component of :math:`\mathbf{D}`
    is a tensor :math:`\mathbf{D}_i\in\mathbb{R}^{D\times D}`. When applied to the :math:`i`-th component of :math:`\nabla\mathbf{f}`, this gives
    the flux

    .. math::
        \Phi_i = \mathbf{D}_i (\nabla\mathbf{f})_i \in \mathbb{R}^D,

    which, applying the divergence, yields

    .. math::
        \Delta f_i = \mathrm{div}(\Phi_i) \in \mathbb{R}.

    In the context of PDE-based image processing, :math:`\Delta f_i` represents the update of :math:`f_i`
    in a denoising/reconstruction process. ``_DiffusionCoefficient`` operators are obtained by
    suitably stacking the tensors :math:`D_i, i=1,\dots, N_0 \cdots N_{D-1}`.

    **Remark 3**

    The class features a ``freeze()`` method. When applied to an array `arr`, it freezes the diffusion coefficient
    at the operator obtained applying ``apply()`` to `arr`.

    The class also features a ``set_frozen_diffusivity()`` method. When fed an operator `frozen_coeff`, it freezes the
    diffusion coefficient at the operator `frozen_coeff`.

    **Remark 4**

    The class features the boolean attribute ``trace_term``, indicating whether the diffusion coefficient is meant to
    be used in a divergence-based (``trace_term`` should be set to `False`) or in a trace-based operator (``trace_term``
    should be set to ``True``). The stacking used to generate the operator in the ``apply()`` method is different in the
    two cases. When ``trace_term`` is ``True``, the output of ``apply()`` is an operator which, when applied to a suitable
    object, already computes the trace of the diffusion tensor applied to that object.

    **Remark 5**

    The class features the attributes ``from_potential`` and ``bounded``. See discussion in
    :py:class:`~pyxu.operator.diffusion._Diffusivity`.

    Developer notes
    --------------
    Currently, instances of _DiffusionCoefficient are not Pyxu operators. This is because the ``apply()`` method returns
    a LinOp/DiagonalOp and not a scalar NDArray. We define some basic arithmetic that allows to consider sums
    between different diffusion coefficient objects and multiplying/dividing by scalars. However, _DiffusionCoefficient
    do not allow multidimensional inputs. Maybe acceptable since it is not a Pyxu operator and these
    operators will likely only ever be used in the context of diffusion processes?
    """

    def __init__(
        self,
        dim_shape: pyct.NDArrayShape,  # dim_shape: pyct.NDArrayShape, codim_shape: pyct.NDArrayShape,
        isotropic: bool = True,
        trace_term: bool = False,
    ):
        r"""
        Parameters
        ----------
        dim_shape: tuple
            Shape of the input array.
        isotropic: bool
            Whether ``_DiffusionCoefficient`` is isotropic or not. Defaults to `True`.
        trace_term: bool
            Whether ``_DiffusionCoefficient`` is meant to be used in a trace formulation or not.
            Defaults to `False`. Method ``apply()`` acts differently depending on value of `trace_term`.

        """
        # self.dim = int(np.prod(dim_shape))
        self.ndims = len(dim_shape) - 1
        self.isotropic = isotropic
        if isotropic:
            super().__init__(dim_shape=dim_shape, codim_shape=(self.ndims, *dim_shape))
        else:
            # might then need to kill first flat dimenstion
            super().__init__(dim_shape=dim_shape, codim_shape=(self.ndims, self.ndims, *dim_shape))
        self.trace_term = trace_term
        self.from_potential = False
        self.frozen = False
        self.frozen_coeff = None
        self.bounded = False
        # compute scaling coefficients for more efficient computation in trace-based case
        self._coeff_op = np.ones((self.ndims, self.ndims), dtype=int)
        if self.trace_term:
            # set extra diagonal coefficients to 2: terms need to be considered twice because of symmetry of both Hessian and diffusion coefficient
            self._coeff_op *= 2
            self._coeff_op -= np.eye(self.ndims, dtype=int)

    def freeze(self, arr: pyct.NDArray):
        if self.frozen:
            warnings.warn("DiffusionCoefficient had already been frozen. Overwriting previous frozen state.")
            self.frozen = False
        self.frozen_coeff = self.apply(arr)
        self.frozen = True

    def set_frozen_coeff(self, frozen_coeff: pyct.OpT):
        if self.frozen:
            warnings.warn("DiffusionCoefficient had already been frozen. Overwriting previous frozen state.")
            self.frozen = False
        self.frozen_coeff = frozen_coeff
        self.frozen = True

    def apply(self, arr: pyct.NDArray) -> pyct.OpT:
        if self.frozen:
            return self.frozen_coeff
        else:
            raise NotImplementedError

    def __call__(self, arr: pyct.NDArray) -> pyct.NDArray:
        """
        Alias for :py:meth:`~pyxu.abc.operator.diffusion._DiffusionCoefficient`.
        """
        return self.apply(arr)


class DiffusionCoeffIsotropic(_DiffusionCoefficient):
    r"""
    Class for isotropic diffusion coefficients, where we follow the definition of isotropy from
    [`Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_].

    **Remark 1**

    *Isotropic* implies that the diffusion tensor is fully described by a diffusivity function :math:`g(\cdot)`.
    Indeed, let :math:`\mathbf{D}` be a ``DiffusionCoeffIsotropic`` and let :math:`f_i` be the :math:`i`-th entry
    (pixel) of the vectorisation of the :math:`D`-dimensional signal

    .. math::
        \mathbf{f} \in \mathbb{R}^{N_{0} \times \cdots \times N_{D-1}}.

    Furthermore, let :math:`(\nabla \mathbf{f})_i = \big((\frac{\partial \mathbf{f}}{\partial x_0})_i,\dots,(\frac{\partial \mathbf{f}}{\partial x_{D-1}})_i\big)^T`.
    We consider the divergence-based case. Then, the :math:`i`-th component of :math:`\mathbf{D}`
    is the tensor :math:`\mathbf{D}_i=(g(\mathbf{f}))_i\,\mathbf{I}_D`, where :math:`(g(\mathbf{f}))_i\in\mathbb{R}` and
    :math:`\mathbf{I}_D` is the :math:`D`-dimensional identity matrix.

    Applying :math:`\mathbf{D}_i` to the :math:`i`-th component of :math:`\nabla\mathbf{f}` gives
    the flux

    .. math::
        \Phi_i = (g(\mathbf{f}))_i\,I_D (\nabla\mathbf{f})_i = (g(\mathbf{f}))_i (\nabla\mathbf{f})_i \in \mathbb{R}^D.

    **Remark 2**

    Instances of :py:class:`~pyxu.operator.diffusion.DiffusionCoeffIsotropic` inherit attributes
    ``from_potential`` and ``bounded`` from the diffusivity.

    Example
    -------

    # .. plot::
    #
    #     import pyxu.operator.linop.diff as pydiff
    #     import pyxu.operator.diffusion as pydiffusion
    #     import skimage as skim
    #
    #     # Import image
    #     image = skim.color.rgb2gray(skim.data.cat())
    #     print(image.shape) #(300, 451)
    #     print(image.size) #135300
    #     # Instantiate gaussian gradient operator
    #     gauss_grad = pydiff.Gradient(dim_shape=(1,*image.shape), directions=(1,2), diff_method="gd", sigma=2)
    #     # Instantiate a diffusivity (e.g., Perona-Malik)
    #     PeronMalik_diffusivity = pydiffusion.PeronaMalikDiffusivity(dim_shape=(1, *image.shape), gradient=gauss_grad, pm_fct="exponential")
    #     # Instantiate an isotropic diffusion coefficient based on the defined diffusivity
    #     PeronaMalik_diffusion_coeff = pydiffusion.DiffusionCoeffIsotropic(dim_shape=(1, *image.shape), diffusivity=PeronMalik_diffusivity)
    #     # Evaluate diffusion coefficient at the image, obtaining an operator of size (2*image.size, 2*image.size).
    #     PMcoeff_eval = PeronaMalik_diffusion_coeff(np.expand_dims(image, 0))
    #     print(PMcoeff_eval) # DiagonalOp(270600, 270600)

    """

    def __init__(self, dim_shape: pyct.NDArrayShape, diffusivity: pyct.OpT = None, trace_term: bool = False):
        r"""

        Parameters
        ----------
        dim_shape: tuple
            Shape of the input array.
        diffusivity: :py:class:`~pyxu.operator.diffusion._Diffusivity`
            Map defining the diffusivity associated to the isotropic coefficient. Defaults to `None`, in which case
            Tikhonov diffusivity is used.
        trace_term: bool
            Whether diffusion coefficient is meant to be used in a trace formulation or not.
            Defaults to `False`. Method ``apply()`` acts differently depending on value of `trace_term`.

        """
        super().__init__(dim_shape=dim_shape, isotropic=True, trace_term=trace_term)
        if diffusivity is None:
            self.diffusivity = TikhonovDiffusivity(dim_shape=dim_shape)
        else:
            msg = "`diffusivity.dim_shape`={} inconsistent with `dim_shape`={}.".format(
                diffusivity.dim_shape, dim_shape
            )
            assert diffusivity.dim_shape == dim_shape, msg
            self.diffusivity = diffusivity
        # self.from_potential = self.diffusivity.from_potential and (not trace_term)
        # if self.diffusivity.bounded:
        #    self.bounded = True

    def apply(self, arr: pyct.NDArray) -> pyct.OpT:
        r"""

        Notes
        -----
        Let :math:`N_{tot}=N_0 \cdots N_{D-1}`, where :math:`D` is the dimension of the signal. The method
        returns an operator which is:

        * if ``trace_term=True`` a :py:class:`~pyxu.abc.operator.LinOp` with shape :math:`(N_{tot}, N_{tot}D)`;
        * if ``trace_term=False`` a :py:class:`~pyxu.operator.linop.base.DiagonalOp` with shape :math:`(N_{tot}D, N_{tot}D)`.

        """
        if not self.frozen:
            xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
            y = self.diffusivity(arr)  # (batch,nchannels,nx,ny)
            if self.trace_term:
                # TODO
                return 0
            else:
                z = [y] * self.ndims
                return xp.stack(z, axis=-4)  # (batch,ndims,nchannels,nx,ny)
                # NEW
                # z = [y]*self.ndims
                # z = xp.hstack(z)
                # now z has shape (batch, stack, 1, nx, ny)
                # do I need an extra dimension though? For anisotropic coefficient, how are we supposed to handle it otherwise?
                # also, should we stack the diffusion coefficient for case when we have multiple channels? otherwise, how can we apply safely our einsum in _diffusion?...
        else:
            return self.frozen_coeff


class _DiffusionCoeffAnisotropic(_DiffusionCoefficient):
    r"""
    Abstract class for anisotropic diffusion coefficients, where we follow the definition of anisotropy from
    [`Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_].

    **Remark 1**

    The class is designed for diffusion tensors obtained as function of the structure tensor (it could be named
    ``_DiffusionCoeffStructTensorBased``). Other types of anisotropic tensors are not meant to be implemented as
    daughter classes of :py:class:`~pyxu.operator.diffusion._DiffusionCoeffAnisotropic`.

    **Remark 2**

    *Anisotropic* implies that the diffusion tensors, locally, are not multiples of the identity matrix.
    Indeed, let :math:`\mathbf{D}` be a ``DiffusionCoeffAnisotropic`` and let :math:`f_i` be the :math:`i`-th entry
    (pixel) of the vectorisation of the :math:`D`-dimensional signal

    .. math::
        \mathbf{f} \in \mathbb{R}^{N_{0} \times \cdots \times N_{D-1}}.

    Furthermore, let :math:`(\nabla \mathbf{f})_i = \big((\frac{\partial \mathbf{f}}{\partial x_0})_i,\dots,(\frac{\partial \mathbf{f}}{\partial x_{D-1}})_i\big)^T`.
    We consider the divergence-based case. Then, the :math:`i`-th component of :math:`\mathbf{D}`
    is the symmetric matrix

    .. math::
        \mathbf{D}_i=
        \left(\begin{array}{ccc}
        (\mathbf{D}_i)_{11} & \cdots & (\mathbf{D}_i)_{1D} \\
        \vdots & \ddots & \vdots \\
        (\mathbf{D}_i)_{1D} & \cdots & (\mathbf{D}_i)_{DD}
        \end{array}\right)\in\mathbb{R}^{D\times D}.

    Applying :math:`\mathbf{D}_i` to the :math:`i`-th component of :math:`\nabla\mathbf{f}` gives
    the flux

    .. math::
        \Phi_i = \mathbf{D}_i (\nabla\mathbf{f})_i \in \mathbb{R}^D,

    which, is not a simple rescaling of the gradient when :math:`\mathbf{D}_i` is not a multiple of the identity matrix
    :math:`\mathbf{I}_D`. Consequently, the flux can point towards directions that are not aligned with gradient, which
    allows for smoothing processes along interesting directions. These directions can be chosen to enhance, for example,
    the `edges` or the `coherence` of the signal.

    **Remark 3**

    As mentioned above, this class considers diffusion coefficients which depend on the structure tensor
    (see :py:class:`~pyxu.operator.linop.filter.StructureTensor`), as we now describe. Let us consider, for each pixel,
    the structure tensor

    .. math::
        \mathbf{S}_i = (\nabla\mathbf{f})_i(\nabla \mathbf{f})_i^T\,\in\mathbb{R}^{D\times D}.

    The matrix :math:`\mathbf{S}_i` is a symmetric positive semidefinite matrix. From its eigenvalue decomposition,
    we obtain the eigenvectors :math:`\mathbf{v}_0,\dots,\mathbf{v}_{D-1}` and the associated sorted eigenvalues
    :math:`e_0 \geq \dots \geq e_{D-1}`, with

    .. math::
        \mathbf{S}_i = \sum_{d=0}^{D-1} e_d\mathbf{v}_d(\mathbf{v}_d)^T.

    The :math:`i`-th component :math:`\mathbf{D}_i` of the ``_DiffusionCoeffAnisotropic`` operator :math:`\mathbf{D}` is
    given by

    .. math::
        \mathbf{D}_i = \sum_{d=0}^{D-1} \lambda_d\mathbf{v}_d(\mathbf{v}_d)^T,

    where the choice of intensities :math:`\lambda_d` for :math:`d=0,\dots,D-1`, which are functions of the eigenvalues
    :math:`e_0, \ldots, e_{D-1}`, specify the ``_DiffusionCoeffAnisotropic`` operator entirely. This results
    in a diffusion coefficient that, when used in the context of diffusion operators, will enhance or dampen features
    by smoothing with different intensities along the different eigenvector directions. More specifically, a large
    intensity value :math:`\lambda_0` will result in a strong diffusion along the direction of the gradient (i.e.,
    edges), whereas :math:`\lambda_d` for :math:`d > 0` controls the diffusion strength in the orthogonal directions.

    **Remark 4**

    Daughter classes of :py:class:`~pyxu.operator.diffusion._DiffusionCoeffAnisotropic` only need to implement the
    method ``_compute_intensities()``, which defines a rule to compute the smoothing intensities  :math:`\lambda_d`
    associated to each eigenvector of the structure tensor. These intensities define the smoothing behavior of the tensor
    (edge-enhancing, coherence-enhancing).

    """

    def __init__(self, dim_shape: pyct.NDArrayShape, structure_tensor: pyct.OpT, trace_term: bool = False):
        """

        Parameters
        ----------
        dim_shape: tuple
            Shape of the input array.
        structure_tensor: :py:class:`~pyxu.operator.linop.filter.StructureTensor`
            Structure tensor operator.
        trace_term: bool
            Whether diffusion coefficient is meant to be used in a trace formulation or not.
            Defaults to `False`. Method ``apply()`` acts differently depending on value of `trace_term`.
        """
        super().__init__(dim_shape=dim_shape, isotropic=False, trace_term=trace_term)
        self.nchannels = dim_shape[0]
        msg = "`structure_tensor.dim_shape`={} inconsistent with `dim_shape`={}.".format(
            structure_tensor.dim_shape, dim_shape[1:]
        )
        assert structure_tensor.dim_shape == dim_shape[1:], msg
        self.structure_tensor = structure_tensor
        # compute the indices of the upper triangular structure tensor to be selected to assemble its full version
        full_matrix_indices = np.zeros((self.ndims, self.ndims), dtype=int)
        upper_matrix_index = 0
        for i in range(self.ndims):
            for j in range(self.ndims):
                if j >= i:
                    full_matrix_indices[i, j] = upper_matrix_index
                    upper_matrix_index += 1
                else:
                    full_matrix_indices[i, j] = full_matrix_indices[j, i]
        self.full_matrix_indices = full_matrix_indices.reshape(-1)  # corresponds to np.array([0,1,1,2]) for 2d image

    # how to enforce precision on tuple of outputs? should I simply use coerce?
    # Actually, both structure_tensor and svd preserve precision, so should not be an issue
    def _eigendecompose_struct_tensor(self, arr: pyct.NDArray) -> (pyct.NDArray, pyct.NDArray):
        r"""
        Notes
        ----
        This function decomposes the structure tensor. For each pixel, the eigenvectors and associated eigenvalues are computed.

        Developer notes
        --------------
        **Remark 1**

        Currently, ``xp.linalg.svd`` is used to decompose the matrices.
        * In NUMPY case, the argument Hermitian=True prompts a call to the efficient ``numpy.linalg.eigh()``.
        * In CUPY case, the argument Hermitian does not exist. There is a method ``cupy.linalg.eigh()`` though, we could leverage it.
        * In DASK case, the argument Hermitian does not exist. Moreover, there is no dask version of ``eigh()``. We should therefore use ``svd()``.


        **Remark 2**

        In the two-dimensional case :math:`D=2`, where the input signal is an image :math:`\mathbf{f}\in\mathbb{R}^{N_0 \times N_1}`, closed-form expressions
        could be used for the eigendecomposition of the structure tensor. To keep things general and be to be able to work in :math:`D` dimensions,
        we do not exploit them and apply ``svd()`` instead.
        """
        xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
        # compute upper/lower triangular component of structure tensor
        # structure_tensor = self.structure_tensor.apply(arr)
        # structure_tensor = xp.sum(structure_tensor, axis=0, keepdims=True)
        # structure_tensor = self.structure_tensor.unravel(structure_tensor).squeeze()
        # structure_tensor = structure_tensor.reshape(structure_tensor.shape[0], -1).T
        structure_tensor = self.structure_tensor.apply(arr)
        # (batch,nchannels,3,nx,ny)
        structure_tensor = xp.sum(
            structure_tensor, axis=-4, keepdims=False
        )  # (batch,3,nx,ny) # summing the info from the three channels together
        # structure_tensor = xp.sum(structure_tensor, axis=-3, keepdims=False)# (batch,nstruct,nx,ny) # if directions passed to structure tensor
        # assemble full structure tensor
        # structure_tensor_full = structure_tensor[:, self.full_matrix_indices].reshape(-1, self.ndims, self.ndims)
        structure_tensor = xp.moveaxis(structure_tensor, -3, -1)  # (batch,nx,ny,3)
        structure_tensor_full = structure_tensor[
            ..., self.full_matrix_indices
        ]  # (batch,nx,ny,4) will work also for multiple batch dimensions!
        structure_tensor_full = structure_tensor_full.reshape(
            *structure_tensor_full.shape[:-1], self.ndims, self.ndims
        )  # (batch,nx,ny,2,2)
        # eigendecompose tensor
        N = pycd.NDArrayInfo
        is_numpy = N.from_obj(arr) == N.NUMPY
        if is_numpy:
            u, e, _ = xp.linalg.svd(structure_tensor_full, hermitian=True)
        else:
            u, e, _ = xp.linalg.svd(structure_tensor_full)
        return u, e  # u->(batch,nx,ny,2,2), e->#(batch,nx,ny,2)

    # @pycrt.enforce_precision(i="eigval_struct")
    def _compute_intensities(self, eigval_struct: pyct.NDArray) -> pyct.NDArray:
        raise NotImplementedError

    # @pycrt.enforce_precision(i=("u", "lambdas"))
    def _assemble_tensors(self, u: pyct.NDArray, lambdas: pyct.NDArray) -> pyct.NDArray:
        # u->(batch,nx,ny,2,2), lambdas->#(batch,nx,ny,2)
        xp = pycu.get_array_module(u)
        diffusion_tensors = xp.zeros_like(u)
        z = (
            lambdas.reshape(*lambdas.shape[:-1], 1, 2) * u
        )  # (batch,nx,ny,2,2), columns of (2,2) matrices are now l1*u1, l2*u2
        for i in range(len(self.dim_shape[1:])):
            # compute rank 1 matrices from eigenvectors, multiply them by intensities and sum up
            diffusion_tensors += z[..., :, i].reshape(*u.shape[:-2], 1, self.ndims) * u[  # (batch,nx,ny,1,2)
                ..., :, i
            ].reshape(
                *u.shape[:-2], self.ndims, 1
            )  # (batch,nx,ny,2,1)  # (batch,nx,ny,2,2)
        # tensors now need to be reshaped for computations in diffusion module: nchannels and matrix ids positions
        diffusion_tensors = xp.moveaxis(diffusion_tensors, [-2, -1], [-4, -3])  # (batch,2,2,nx,ny)
        diffusion_tensors = xp.stack([diffusion_tensors] * self.nchannels, -3)  # (batch,2,2,nchannels,nx,ny)
        return diffusion_tensors

    def apply(self, arr: pyct.NDArray) -> pyct.OpT:
        r"""

        Notes
        -----
        Let :math:`N_{tot}=N_0\cdots N_{D-1}`, where :math:`D` is the dimension of the signal. The number of
        supradiagonal elements in a :math:`\mathbb{R}^{D\times D}` matrix is :math:`D_{extra}=D(D+1)/2`. The method
        returns an operator:

        * if ``trace_term=True``, a :py:class:`~pyxu.abc.operator.LinOp` with shape :math:`(N_{tot}, N_{tot}D_{extra})`;
        * if ``trace_term=False``, a :py:class:`~pyxu.operator.linop.base.DiagonalOp` with shape :math:`(N_{tot}D, N_{tot}D)`.

        **Remark**

        The current implementation in the trace-based case (``trace_term=True``) relies on the fact that, for each pixel,
        both the Hessian and the diffusion tensor are symmetric.
        """
        if not self.frozen:
            u, e = self._eigendecompose_struct_tensor(arr)
            lambdas = self._compute_intensities(e)
            tensors = self._assemble_tensors(u, lambdas)
            return tensors  # (batch,2,2,nchannels,nx,ny)
        else:
            return self.frozen_coeff


class DiffusionCoeffAnisoEdgeEnhancing(_DiffusionCoeffAnisotropic):
    r"""
    Edge-enhancing anisotropic diffusion coefficient, based on the :py:class:`~pyxu.operator.linop.filter.StructureTensor`
    [see `Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_].

    Notes
    -----

    Let us consider the two-dimensional case :math:`D=2`, where the input signal is an image :math:`\mathbf{f}\in\mathbb{R}^{N_0 \times N_1}`.
    We follow the notation from the documentation of :py:class:`~pyxu.operator.diffusion._DiffusionCoeffAnisotropic`. In the context of
    diffusion operators, operators of :py:class:`~pyxu.operator.diffusion.DiffusionCoeffAnisoEdgeEnhancing` can be
    used to enhance the edges in the image.
    Let us consider the :math:`i`-th pixel of the image. The edge-enhancing effect is achieved with the following choice
    of smoothing intensities associated to the eigenvalues of the structure tensor :math:`\mathbf{S}_i`:

    .. math::
        \lambda_0 &= g(e_0),\\
        \lambda_1 &= 1,

    with

    .. math::
        g(e_0) :=
       \begin{cases}
           1 & \text{if } e_0 \leq 0 \\
           1 - \exp\big(\frac{-C}{(e_0/\beta)^m}\big) & \text{if } e_0 >0,
       \end{cases}

    where :math:`\beta` is a contrast parameter, :math:`m` controls the decay rate of :math:`\lambda_0` as a function
    of :math:`e_0`, and :math:`C\in\mathbb{R}` is a constant.

    Since :math:`\lambda_1 \geq \lambda_0`, the smoothing intensity is stronger in the direction of the second
    eigenvector of :math:`\mathbf{S}_i` (perpendicular to the gradient). Moreover, :math:`\lambda_0` is a decreasing
    function of :math:`e_0`, which indicates that when the gradient magnitude is high (sharp edges), there is little
    smoothing in the direction of the gradient, i.e., edges are preserved.

    **Remark 1**

    Currently, only two-dimensional case :math:`D=2` is handled. Need to implement rules to compute intensity for case :math:`D>2`.

    **Remark 2**

    Performance of the method can be quite sensitive to the hyperparameters :math:`\beta, m`, particularly :math:`\beta`.

    Example
    -------

    .. plot::

        import numpy as np
        import pyxu.operator.linop.filter as pyfilt
        import pyxu.operator.diffusion as pydiffusion
        import skimage as skim
        # Import image
        image = skim.color.rgb2gray(skim.data.cat())
        print(image.shape) #(300, 451)
        print(image.size) #135300
        # Instantiate structure tensor
        structure_tensor = pyfilt.StructureTensor(dim_shape=image.shape, diff_method="gd", smooth_sigma=0,
                                                  mode="symmetric", sigma=2)
        # Instantiate diffusion coefficient
        EdgeEnhancing_coeff = pydiffusion.DiffusionCoeffAnisoEdgeEnhancing(dim_shape=image.shape, structure_tensor=structure_tensor)
        # Evaluate diffusion coefficient at the image, obtaining an operator of size (2*image.size, 2*image.size).
        EdheEnhance_coeff_eval = EdgeEnhancing_coeff(image.reshape(1,-1))
        print(EdheEnhance_coeff_eval) # SquareOp(270600, 270600)

    """

    def __init__(
        self,
        dim_shape: pyct.NDArrayShape,
        structure_tensor: pyct.OpT,
        trace_term: bool = False,
        beta: pyct.Real = 1.0,
        m: pyct.Real = 4.0,
    ):
        """

        Parameters
        ----------
        dim_shape: tuple
            Shape of the input array.
        structure_tensor: :py:class:`~pyxu.operator.linop.filter.StructureTensor`
            Structure tensor operator.
        trace_term: bool
            Whether diffusion coefficient is meant to be used in a trace formulation or not.
            Defaults to `False`. Method ``apply()`` acts differently depending on value of `trace_term`.
        beta: pyct.Real
            Contrast parameter. Defaults to `1`.
        m: pyct.Real
            Decay rate in intensity expression. Defaults to `4`.
        """
        # assert len(dim_shape) == 2, "`dim_shape` has more than two dimensions, not handled yet"
        super().__init__(dim_shape=dim_shape, structure_tensor=structure_tensor, trace_term=trace_term)
        assert beta > 0, "contrast parameter `beta` must be strictly positive"
        self.beta = beta
        assert m > 0, "decay rate `m` must be strictly positive"
        self.m = 4
        self.bounded = True

        # compute normalization constant c [see `Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_]
        def f(c: pyct.Real, m: pyct.Integer):
            return 1 - np.exp(-c) * (1 + 2 * m * c)

        self.c = sciop.brentq(functools.partial(f, m), 1e-2, 100)

    # @pycrt.enforce_precision(i="eigval_struct")
    def _compute_intensities(self, eigval_struct: pyct.NDArray) -> pyct.NDArray:
        # (batch,nx,ny,2)
        xp = pycu.get_array_module(eigval_struct)
        lambdas = xp.ones(eigval_struct.shape, dtype=eigval_struct.dtype)
        nonzero_contrast_locs = ~xp.isclose(eigval_struct[..., 0], 0)
        # Inplace implementation of
        #   lambdas[nonzero_contrast_locs, 0] = 1 - xp.exp(- self.c / ((eigval_struct[nonzero_contrast_locs, 0] / self.beta) ** self.m))
        lambda0_nonzerolocs = eigval_struct[nonzero_contrast_locs, 0]
        lambda0_nonzerolocs /= self.beta**2
        lambda0_nonzerolocs **= -self.m
        lambda0_nonzerolocs *= -self.c
        lambda0_nonzerolocs = -xp.exp(lambda0_nonzerolocs)
        lambda0_nonzerolocs += 1
        lambdas[nonzero_contrast_locs, 0] = lambda0_nonzerolocs
        return lambdas


class DiffusionCoeffAnisoCoherenceEnhancing(_DiffusionCoeffAnisotropic):
    r"""
    Coherence-enhancing anisotropic diffusion coefficient, based on structure tensor [see `Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_].

    Notes
    -----

    Let us consider the two-dimensional case :math:`D=2`, where the input signal is an image :math:`\mathbf{f}\in\mathbb{R}^{N_0 \times N_1}`.
    We follow the notation from the documentation of :py:class:`~pyxu.operator.diffusion._DiffusionCoeffAnisotropic`. In the context of
    diffusion operators, operators of :py:class:`~pyxu.operator.diffusion.DiffusionCoeffAnisoCoherenceEnhancing` can be
    used to enhance the coherence in the image.
    Let us consider the :math:`i`-th pixel of the image. The coherence enhancing effect is achieved with the following choice
    of smoothing intensities associated to the eigenvalues of the structure tensor :math:`\mathbf{S}_i`:

    .. math::
        \lambda_0 &= \alpha,\\
        \lambda_1 &= h(e_0, e_1),

    with

    .. math::
        h(e_0, e_1) :=
       \begin{cases}
           \alpha & \text{if } e_0=e_1 \\
           \alpha + (1-\alpha) \exp \big(\frac{-C}{(e_0-e_1)^{2m}}\big) & \text{otherwise},
       \end{cases}

    where :math:`\alpha \in (0, 1)` controls the smoothing intensity in the first eigendirection, :math:`m` controls the
    decay rate of :math:`\lambda_0` as a function of :math:`(e_0-e_1)`, and :math:`C\in\mathbb{R}` is a constant.

    For regions with low coherence, which is measured as :math:`(e_0-e_1)^2`, smoothing is performed uniformly along all
    directions with intensity :math:`\lambda_1 \approx \lambda_0 = \alpha`. For regions with high coherence, we have
    :math:`\lambda_1 \approx 1 > \lambda_0 = \alpha`, hence the smoothing intensity is higher in the direction of the
    gradient.

    **Remark 1**

    Currently, only the two-dimensional case :math:`D=2` is handled. Need to implement rules to compute intensity for case :math:`D>2`.

    **Remark 2**

    The performance of the method can be quite sensitive to the hyperparameters :math:`\alpha, m`, particularly :math:`\alpha`.

    Example
    -------

    .. plot::

        import numpy as np
        import pyxu.operator.linop.filter as pyfilt
        import pyxu.operator.diffusion as pydiffusion
        import skimage as skim
        # Import image
        image = skim.color.rgb2gray(skim.data.cat())
        print(image.shape) #(300, 451)
        print(image.size) #135300
        # Instantiate structure tensor
        structure_tensor = pyfilt.StructureTensor(dim_shape=image.shape, diff_method="gd", smooth_sigma=0,
                                                  mode="symmetric", sigma=2)
        # Instantiate diffusion coefficient
        CoherenceEnhancing_coeff = pydiffusion.DiffusionCoeffAnisoCoherenceEnhancing(dim_shape=image.shape, structure_tensor=structure_tensor)
        # Evaluate diffusion coefficient at the image, obtaining an operator of size (2*image.size, 2*image.size).
        CoherenceEnhance_coeff_eval = CoherenceEnhancing_coeff(image.reshape(1,-1))
        print(CoherenceEnhance_coeff_eval) # SquareOp(270600, 270600)

    """

    def __init__(
        self, dim_shape: pyct.NDArrayShape, structure_tensor: pyct.OpT, trace_term: bool = False, alpha=0.1, m=1
    ):
        """

        Parameters
        ----------
        dim_shape: tuple
            Shape of the input array.
        structure_tensor: :py:class:`~pyxu.operator.linop.filter.StructureTensor`
            Structure tensor operator.
        trace_term: bool
            Whether diffusion coefficient is meant to be used in a trace formulation or not.
            Defaults to `False`. Method ``apply()`` acts differently depending on value of `trace_term`.
        alpha: pyct.Real
            Smoothing intensity in first eigendirection. Defaults to `0.1`.
        m: pyct.Real
            Decay rate in intensity expression. Defaults to `1`.
        """
        # assert len(dim_shape) == 2, "`dim_shape` has more than two dimensions, not handled yet"
        super().__init__(dim_shape=dim_shape, structure_tensor=structure_tensor, trace_term=trace_term)
        assert alpha > 0, "intensity parameter `alpha` must be strictly positive"
        self.alpha = alpha
        assert m > 0, "decay rate `m` must be strictly positive"
        self.m = m
        # constant C set to 1 for now
        self.c = 1.0
        self.bounded = True

    # @pycrt.enforce_precision(i="eigval_struct")
    def _compute_intensities(self, eigval_struct: pyct.NDArray) -> pyct.NDArray:
        # (batch,nx,ny,2)
        xp = pycu.get_array_module(eigval_struct)
        coherence = (eigval_struct[..., 0] - eigval_struct[..., 1]) ** 2
        lambdas = self.alpha * xp.ones(eigval_struct.shape, dtype=eigval_struct.dtype)
        nonzero_coherence_locs = ~xp.isclose(coherence, 0)  # xp.isclose!!! it was np.isclose before
        # Inplace implementation of
        #   lambdas[nonzero_coherence_locs, 1] = self.alpha + (1-self.alpha)*np.exp(-1./(coherence[nonzero_coherence_locs] ** self.m))
        lambda1_nonzerolocs = coherence[nonzero_coherence_locs]
        lambda1_nonzerolocs **= -(self.m)
        lambda1_nonzerolocs *= -self.c
        lambda1_nonzerolocs = xp.exp(lambda1_nonzerolocs)
        lambda1_nonzerolocs *= 1 - self.alpha
        lambda1_nonzerolocs += self.alpha
        lambdas[nonzero_coherence_locs, 1] = lambda1_nonzerolocs
        return lambdas


class DiffusionCoeffAnisotropic(_DiffusionCoeffAnisotropic):
    def __init__(self, dim_shape: pyct.NDArrayShape, structure_tensor: pyct.OpT, trace_term: bool = False, alpha=0.1):
        """

        Parameters
        ----------
        dim_shape: tuple
            Shape of the input array.
        structure_tensor: :py:class:`~pyxu.operator.linop.filter.StructureTensor`
            Structure tensor operator.
        trace_term: bool
            Whether diffusion coefficient is meant to be used in a trace formulation or not.
            Defaults to `False`. Method ``apply()`` acts differently depending on value of `trace_term`.
        alpha: pyct.Real
            Smoothing intensity in first eigendirection. Defaults to `0.1`.
        m: pyct.Real
            Decay rate in intensity expression. Defaults to `1`.
        """
        # assert len(dim_shape) == 2, "`dim_shape` has more than two dimensions, not handled yet"
        super().__init__(dim_shape=dim_shape, structure_tensor=structure_tensor, trace_term=trace_term)
        assert alpha > 0, "intensity parameter `alpha` must be strictly positive"
        self.alpha = alpha
        self.bounded = True

    # @pycrt.enforce_precision(i="eigval_struct")
    def _compute_intensities(self, eigval_struct: pyct.NDArray) -> pyct.NDArray:
        # (batch,nx,ny,2)
        xp = pycu.get_array_module(eigval_struct)
        lambdas = xp.ones(eigval_struct.shape, dtype=eigval_struct.dtype)
        lambdas[..., 0] = self.alpha
        return lambdas
