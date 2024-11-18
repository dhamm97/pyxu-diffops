import warnings

import numpy as np
import pyxu.abc as pyca
import pyxu.info.deps as pycd
import pyxu.info.ptype as pyct
import pyxu.operator.linop.diff as pydiff
import pyxu.util as pycu
import scipy.sparse as sp

import pyxu_diffops.operator.diffusion._diffusion_coeff as diffcoeff

# import cupyx.scipy.sparse as csp

__all__ = [
    "_Diffusion",
]


class _Diffusion(pyca.DiffFunc):
    r"""
    Abstract class for diffusion operators
    [see `Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_ and `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_].

    Notes
    -----

    This class provides an interface to deal with PDE-based regularisation. For simplicity, throughout the
    documentation we consider a :math:`2`-dimensional signal :math:`\mathbf{f} \in \mathbb{R}^{N_{0} \times N_1}`,
    but higher dimensional signals could be considered. We denote by :math:`f_i` the :math:`i`-th entry (pixel)
    of the vectorisation of :math:`\mathbf{f}`, :math:`i=0,\dots,(N_0N_1-1)`. Furthermore, let
    :math:`(\nabla \mathbf{f})_i = \big((\frac{\partial \mathbf{f}}{\partial x_0})_i,\dots,(\frac{\partial \mathbf{f}}{\partial x_{D-1}})_i\big)^T`.

    To give some intuition, let us first consider a simple case: Tikhonov regularisation, corresponding to linear, isotropic,
    homogeneous smoothing. This yields the regularisation functional

    .. math::
        \phi(\mathbf{f}) = \frac{1}{2}\Vert \nabla \mathbf{f} \Vert_2^2
                         = \frac{1}{2}\sum_{i=0}^{N_0 N_1-1} \Vert (\nabla \mathbf{f})_i \Vert_2^2.

    Then, we have

    .. math::
        \nabla \phi(\mathbf{f}) = \nabla^T\nabla\mathbf{f}
                                = -\mathrm{div}(\nabla\mathbf{f})

    where :math:`\nabla^T` is the adjoint of the gradient operator and where we exploited the fact that
    :math:`\nabla^T = -\mathrm{div}`, the divergence. To solve the optimization problem

    .. math::
        \underset{\mathbf{f}}{\mathrm{argmin}} \ \phi(\mathbf{f}),

    we could apply gradient descent, whose update formula is given by

    .. math::
        \mathbf{f}_{n+1} = \mathbf{f}_n + \eta \mathrm{div}(\nabla\mathbf{f}_n),

    where :math:`\mathbf{f}_n` is the :math:`n`-th iterate and :math:`\eta` is the step size of the algorithm. The above
    update equation can be interpreted as one step in time of the explicit Euler integration method applied to the PDE

    .. math::
        \frac{\partial\mathbf{f}}{\partial t} = \mathrm{div}(\nabla\mathbf{f})
                                              = \Delta \mathbf{f},

    with the step size :math:`\Delta t=\eta` in time.
    This time-dependent PDE represents the gradient flow formulation of the original optimization problem, where the
    time is artificial and characterises the optimization process. We recognise, moreover, that we obtain the
    well-known heat equation.

    We can thus let the PDE evolve in time until it reaches a steady state :math:`\frac{\partial\mathbf{f}}{\partial t}=0`.
    The solution will therefore satisfy the first order optimality condition :math:`\nabla \phi(\mathbf{f})=0`.

    If formulated as above, a trivial steady state corresponding to an infinitely flat solution will be obtained.
    However, if the functional :math:`\phi(\cdot)` is combined with a data-fidelity functional :math:`\ell(\cdot)`
    in the context of an inverse problem, an extra term :math:`\nabla \ell(\cdot)` will arise in the gradient flow
    formulation. This will lead to a non-trivial steady state corresponding to the balance between the data-fidelity
    and regularisation terms.

    In the context of PDE-based regularisation, it is not necessary to limit ourselves to consider cases where it is possible
    to explicitly define a variational functional :math:`\phi(\cdot)`. In the spirit of Plug&Play (PnP) approaches,
    we can consider diffusion operators that are only characterised by their smoothing action in gradient flow form: no
    underlying functional :math:`\phi(\cdot)` may exist. This allows to study complex diffusion processes designed to
    enhance specific features of the image.

    In particular, we consider diffusion processes that, in their most general form, can be written as the composite term

    .. math::
        \frac{\partial\mathbf{f}}{\partial t} = \mathbf{D}_{out}\mathrm{div}(\mathbf{D}_{in}\nabla\mathbf{f})
        + \mathbf{b} + \mathbf{T}_{out}\mathrm{trace}\big(\mathbf{T}_{in}\mathbf{H}(\mathbf{f})\big) + (\nabla\mathbf{f})^T \mathbf{J}_{\mathbf{w}}\mathbf{w},

    where
        * :math:`\mathbf{D}_{out} = \mathbf{D}_{out}(\mathbf{f}) \in \mathbb{R}^{N_{tot} \times N_{tot} }` is the outer diffusivity for the divergence term;
        * :math:`\mathbf{D}_{in} = \mathbf{D}_{in}(\mathbf{f}) \in \mathbb{R}^{D N_{tot} \times D N_{tot} }` is the diffusion coefficient for the divergence term;
        * :math:`\mathbf{b} = \mathbf{b}(\mathbf{f}) \in \mathbb{R}^{N_{tot}}` is the balloon force;
        * :math:`\mathbf{T}_{out} = \mathbf{T}_{out}(\mathbf{f}) \in \mathbb{R}^{N_{tot} \times N_{tot} }` is the outer diffusivity for the trace term;
        * :math:`\mathbf{T}_{in} = \mathbf{T}_{in}(\mathbf{f}) \in \mathbb{R}^{D^2 N_{tot} \times D^2 N_{tot} }` is the diffusion coefficient for the trace term;
        * :math:`\mathbf{w} \in \mathbb{R}^{D N_{tot}}` is a vector field assigning a :math:`2`-dimensional vector to each pixel;
        * :math:`\mathbf{J}_\mathbf{w} \in \mathbb{R}^{D N_{tot}^2 \times D N_{tot}}` is the Jacobian of the vector field :math:`\mathbf{w}`.

    The right-hand side of the above PDE represents the output of the ``grad()`` method applied to the image :math:`\mathbf{f}`.

    To conclude, we remark that the effect of the diffusion operator on an image :math:`\mathbf{f}` can be better understood
    by focusing on a single pixel :math:`f_i` of the vectorisation of :math:`\mathbf{f}` (see, e.g., discussion in
    :py:class:`~pyxu.operator.diffusion._DiffusionCoefficient`).

    **Remark 1**

    ``_DiffusionOp`` is an atypical :py:class:`~pyxu.abc.operator.ProxDiffFunc`. Indeed,
    the ``apply()`` method is not necessarily defined, in the case of implicitly defined functionals.
    The key method is ``grad()``, necessary to perform gradient flow optimization (and also used by
    ``prox()``).

    **Remark 2**

    The ``apply()`` method raises a ``NotImplementedError`` unless the diffusion term is known to derive from
    a variational formulation. Currently, only the case where :math:`\mathbf{D}_{in}` is a
    :py:class:`~pyxu.operator.diffusion.DiffusionCoeffIsotropic` and all other diffusion coefficients/diffusivities
    are `None` may detect an underlying variational formulation. Other cases exist but are not treated for now.

    **Remark 3**

    This class is not meant to be directly used, hence the underscore ``_DiffusionOp`` signalling it is private.
    In principle, users should rely on the daughter classes :py:class:`~pyxu.operator.diffusion.DivergenceDiffusionOp`,
    :py:class:`~pyxu.operator.diffusion.SnakeDiffusionOp`, :py:class:`~pyxu.operator.diffusion.TraceDiffusionOp`,
    :py:class:`~pyxu.operator.diffusion.CurvaturePreservingDiffusionOp`.


    Developer Notes
    ---------------
    * In method ``grad()``, to avoid using the @vectorize decorator, all ``_compute()`` functions should be changed, suitably stacking
      all operators and results along the stacking dimensions. For now this has not been done, to be discussed if less naif
      vectorisation is important. It would be cumbersome especially for the terms involving diffusion coefficients, whose
      ``apply()`` method returns a Pyxu operator.

    * For now, user is meant to initialize independently all the building blocks and provide them at initialization of a
      diffusion operator. We could provide, of course, simpler interfaces for some of the most standard diffusion operators.
      Still, even in current form, module should hopefully be relatively simple to use when provided with some examples.
    """

    def __init__(
        self,
        dim_shape: pyct.NDArrayShape,
        gradient: pyct.OpT = None,
        hessian: pyct.OpT = None,
        outer_diffusivity: pyct.OpT = None,
        diffusion_coefficient: pyct.OpT = None,
        extra_diffusion_term: pyct.OpT = None,
        outer_trace_diffusivity: pyct.OpT = None,
        trace_diffusion_coefficient: pyct.OpT = None,
        matrix_based_impl: bool = False,
    ):
        r"""

        Parameters
        ----------
        dim_shape: pyct.Shape
            Shape of the input array.
        gradient: :py:class:`~pyxu.operator.linop.diff.Gradient`
            Gradient operator. Defaults to `None`.
        hessian:  :py:class:`~pyxu.operator.linop.diff.Hessian`
            Hessian operator. Defaults to `None`.
        outer_diffusivity: :py:class:`~pyxu.operator.diffusion._Diffusivity`
            Outer diffusivity operator of the divergence term.
        diffusion_coefficient: :py:class:`~pyxu.operator.diffusion._DiffusionCoefficient`
            Diffusion coefficient operator of the divergence term.
        balloon_force: :py:class:`~pyxu.operator.diffusion._BalloonForce`
            Balloon force operator.
        outer_trace_diffusivity: :py:class:`~pyxu.operator.diffusion._Diffusivity`
            Outer diffusivity operator of the trace term.
        trace_diffusion_coefficient: :py:class:`~pyxu.operator.diffusion._DiffusionCoefficient`
            Diffusion coefficient operator of the trace term.
        curvature_preservation_field: pyct.NDArray
            Vector field along which curvature should be preserved. Defaults to `None`.
        prox_sigma: pyct.Real
            Size of the structures that should be smoothed out by prox computation (when implemented in PnP-fashion).

        Notes
        ----
        The parameter ``prox_sigma`` is expressed in pixel units. Independently of the ``sampling`` of the differential
        operators involved, ``prox_sigma`` equal to `1` is meant to yield a prox which smoothes structures of the order
        of one pixel.
        """
        self.ndims = len(dim_shape) - 1
        self.nchannels = dim_shape[0]
        self.num_pixels = np.prod(dim_shape[1:])
        super().__init__(dim_shape=dim_shape, codim_shape=1)
        # super().__init__(shape=(1, int(np.prod(dim_shape))))
        # sanitize inputs
        (
            gradient,
            hessian,
            diffusion_coefficient,
            trace_diffusion_coefficient,
            sampling,
        ) = self._sanitize_init_args(
            dim_shape=dim_shape,
            gradient=gradient,
            hessian=hessian,
            outer_diffusivity=outer_diffusivity,
            diffusion_coefficient=diffusion_coefficient,
            extra_diffusion_term=extra_diffusion_term,
            outer_trace_diffusivity=outer_trace_diffusivity,
            trace_diffusion_coefficient=trace_diffusion_coefficient,
            matrix_based_impl=matrix_based_impl,
        )
        self.outer_diffusivity = outer_diffusivity
        self.diffusion_coefficient = diffusion_coefficient
        if diffusion_coefficient is not None:
            if diffusion_coefficient.isotropic:
                self.einsum_subs = "i...,i...->i..."
            else:
                self.einsum_subs = "...ijklm,...jklm->...iklm"
        self.extra_diffusion_term = extra_diffusion_term
        self.outer_trace_diffusivity = outer_trace_diffusivity
        self.trace_diffusion_coefficient = trace_diffusion_coefficient
        if trace_diffusion_coefficient is not None or outer_trace_diffusivity is not None:
            # compute the indices of the upper triangular hessian to be selected to assemble its full version
            full_matrix_indices = np.zeros((self.ndims, self.ndims), dtype=int)
            upper_matrix_index = 0
            for i in range(self.ndims):
                for j in range(self.ndims):
                    if j >= i:
                        full_matrix_indices[i, j] = upper_matrix_index
                        upper_matrix_index += 1
                    else:
                        full_matrix_indices[i, j] = full_matrix_indices[j, i]
            self.full_matrix_indices = full_matrix_indices.reshape(
                -1
            )  # corresponds to np.array([0,1,1,2]) for 2d image
        self.sampling = sampling
        self.gradient = gradient
        self.hessian = hessian
        self.matrix_based_impl = matrix_based_impl
        if matrix_based_impl:
            self._assemble_matrix_based()
        # set lipschitz and diff_lipschitz to np.inf
        self.lipschitz = np.inf
        self.diff_lipschitz = np.inf

    def _sanitize_init_args(
        self,
        dim_shape: pyct.NDArrayShape,
        gradient: pyct.OpT,
        hessian: pyct.OpT,
        outer_diffusivity: pyct.OpT,
        diffusion_coefficient: pyct.OpT,
        extra_diffusion_term: pyct.OpT,
        outer_trace_diffusivity: pyct.OpT,
        trace_diffusion_coefficient: pyct.OpT,
        matrix_based_impl: bool = False,
    ):
        # if hessian is not None:
        #     nb_upper_entries = round(self.ndims * (self.ndims + 1) / 2)
        #     expected_codim = nb_upper_entries * self.dim
        #     assert hessian.codim == expected_codim, '`hessian` expected to be initialized with `directions`="all"'

        if outer_diffusivity is not None and diffusion_coefficient is None:
            diffusion_coefficient = diffcoeff.DiffusionCoeffIsotropic(dim_shape=dim_shape)
            msg = "No`diffusion_coefficient` was passed, initializing to Tikhonov isotropic coefficient."
            warnings.warn(msg)

        if outer_trace_diffusivity is not None and trace_diffusion_coefficient is None:
            trace_diffusion_coefficient = diffcoeff.DiffusionCoeffIsotropic(dim_shape=dim_shape)
            msg = "No`trace_diffusion_coefficient` was passed, initializing to Tikhonov isotropic coefficient."
            warnings.warn(msg)

        if (diffusion_coefficient is None) and (extra_diffusion_term is None) and (trace_diffusion_coefficient is None):
            msg = "\n".join(
                [
                    "Cannot instantiate the diffusion operator. Pass at least one of the following:",
                    "`diffusion_coefficient`, `extra_diffusion_term`, `trace_diffusion_coefficient`.",
                ]
            )
            raise ValueError(msg)

        if diffusion_coefficient is not None and gradient is None:
            msg = "\n".join(
                [
                    "No`gradient` was passed, needed for divergence term involving `diffusion_coefficient`.",
                    "Initializing a forward finite difference operator with unitary sampling as default.",
                ]
            )
            warnings.warn(msg)
            gradient = pydiff.Gradient(
                dim_shape=dim_shape,
                directions=(1, 2),
                diff_method="fd",
                sampling=1.0,
                mode="symmetric",
                scheme="forward",
            )

        if trace_diffusion_coefficient is not None and hessian is None:
            msg = "\n".join(
                [
                    "No `hessian` was passed, needed for trace term involving `trace_diffusion_coefficient`.",
                    "Initializing a central finite difference operator with unitary sampling as default.",
                ]
            )
            warnings.warn(msg)
            hessian = pydiff.Hessian(
                dim_shape=dim_shape, diff_method="fd", mode="symmetric", sampling=1.0, scheme="central", accuracy=2
            )

        # check dim_shapes consistency
        _to_be_checked = {
            "outer_diffusivity": outer_diffusivity,
            "diffusion_coefficient": diffusion_coefficient,
            "extra_diffusion_term": extra_diffusion_term,
            "outer_trace_diffusivity": outer_trace_diffusivity,
            "trace_diffusion_coefficient": trace_diffusion_coefficient,
            "gradient": gradient,
            # "hessian": hessian,
        }
        for i in _to_be_checked:
            if _to_be_checked[i]:
                msg = "`{}.dim_shape`=({}) inconsistent with `dim_shape`={}.".format(
                    i, _to_be_checked[i].dim_shape, dim_shape
                )
                assert _to_be_checked[i].dim_shape == dim_shape, msg

        # check sampling consistency
        _to_be_checked = {}
        if gradient is not None:
            _to_be_checked["`gradient`"] = gradient.meta.sampling[0]
        # if hessian is not None:
        #     _to_be_checked["`hessian`"] = hessian.meta.sampling[0]
        if extra_diffusion_term is not None:
            if extra_diffusion_term.gradient:
                _to_be_checked["`extra_diffusion_term.gradient`"] = extra_diffusion_term.gradient.meta.sampling[0]
        if outer_diffusivity is not None:
            if outer_diffusivity.gradient is not None:
                _to_be_checked["`outer_diffusivity.gradient`"] = outer_diffusivity.gradient.meta.sampling[0]
        if outer_trace_diffusivity is not None:
            if outer_trace_diffusivity.gradient is not None:
                _to_be_checked["`outer_trace_diffusivity.gradient`"] = outer_trace_diffusivity.gradient.meta.sampling[0]
        if diffusion_coefficient is not None:
            if diffusion_coefficient.isotropic:
                if diffusion_coefficient.diffusivity.gradient is not None:
                    _to_be_checked["`diffusion_coefficient.diffusivity.gradient`"] = (
                        diffusion_coefficient.diffusivity.gradient.meta.sampling[0]
                    )
            # else:
            #     if diffusion_coefficient.structure_tensor is not None:
            #         _to_be_checked[
            #             "`diffusion_coefficient.structure_tensor.gradient`"
            #         ] = diffusion_coefficient.structure_tensor.grad.meta.sampling[0]
        if trace_diffusion_coefficient is not None:
            if trace_diffusion_coefficient.isotropic:
                if trace_diffusion_coefficient.diffusivity.gradient is not None:
                    _to_be_checked["`trace_diffusion_coefficient.diffusivity.gradient`"] = (
                        trace_diffusion_coefficient.diffusivity.gradient.meta.sampling[0]
                    )
            # else:
            #     if trace_diffusion_coefficient.structure_tensor is not None:
            #         _to_be_checked[
            #             "`trace_diffusion_coefficient.structure_tensor.gradient`"
            #         ] = trace_diffusion_coefficient.structure_tensor.grad.meta.sampling[0]
        if _to_be_checked:
            s0 = list(_to_be_checked.values())[0]
            op_base = list(_to_be_checked.keys())[0]
            for s in _to_be_checked:
                assert _to_be_checked[s] == s0, "Inconsistent `sampling` for differential operators {} and {}.".format(
                    op_base, s
                )
            sampling = s0
        else:
            sampling = None

        if matrix_based_impl:
            # assert dim_shape[0] == 1, "Matrix-based implementation does not support multichannel images."
            assert len(self.dim_shape[1:]) == 2, "Only 2d images currently accepted for matrix-based implementation."
            msg = "Matrix-based implementation currently only available for divergence-based diffusion operators."
            assert (
                trace_diffusion_coefficient is None and outer_trace_diffusivity is None and extra_diffusion_term is None
            ), msg
            assert (
                diffusion_coefficient.frozen
            ), "Parameter `diffusion_coefficient` must be frozen for matrix-based implementation."

        # if trace_diffusion_coefficient is isotropic,
        # convert hessian to second derivative operator
        # if trace_diffusion_coefficient is not None:
        #     if trace_diffusion_coefficient.isotropic:
        #         ops = []
        #         idx = 0
        #         for dim in range(self.ndims):
        #             # select second order derivative operators
        #             ops.append(hessian._block[(idx, 0)])
        #             idx += self.ndims - dim
        #         hessian = pyblock.vstack(ops)
        #         hessian = pydiff._make_unravelable(hessian, dim_shape=dim_shape)

        # returning only objects that might have been modified.
        return (
            gradient,
            hessian,
            diffusion_coefficient,
            trace_diffusion_coefficient,
            sampling,
        )

    def _assemble_matrix_based(self):
        # VERY CAREFUL WITH DIFF_LIPSCHITZ! IF I WANT TO USE PREVIOUS ONE, I NEED TO DIVIDE Dx, Dy BY SAMPLING, right?
        # currently implemented with this division. monitor situatino and remember.

        # Assemble matrix-based version of the operator.
        # Can lead to significant speed-up for frozen diffusion coefficients when input size is fairly small.
        xp = pycu.get_array_module(self.diffusion_coefficient.frozen_coeff)
        N = pycd.NDArrayInfo
        is_numpy = N.from_obj(self.diffusion_coefficient.frozen_coeff) == N.NUMPY
        is_cupy = N.from_obj(self.diffusion_coefficient.frozen_coeff) == N.CUPY
        Dx = -xp.diag(xp.ones(self.dim_shape[1]) / self.sampling[1]) + xp.diag(
            xp.ones(self.dim_shape[1] - 1) / self.sampling[1], 1
        )
        Dx[-1, -1] = 0  # symmetric boundary conditions, no flux
        Dy = -xp.diag(xp.ones(self.dim_shape[2]) / self.sampling[2]) + xp.diag(
            xp.ones(self.dim_shape[1] - 1) / self.sampling[2], 1
        )
        Dy[-1, -1] = 0  # symmetric boundary conditions, no flux
        # define gradient matrix
        D = xp.vstack((xp.kron(Dx, xp.eye(self.dim_shape[2])), xp.kron(xp.eye(self.dim_shape[1]), Dy)))
        # assemble diffusion tensor as full matrix
        diff_coeff_tensor = self.diffusion_coefficient.frozen_coeff  # (2,2,nchannels=1,nx,ny)
        diff_coeff_tensor = diff_coeff_tensor.squeeze()  # (2,2,nx,ny)
        W = (
            xp.diag(xp.hstack((diff_coeff_tensor[0, 0, :, :].flatten(), diff_coeff_tensor[1, 1, :, :].flatten())))
            + xp.diag(diff_coeff_tensor[0, 1, :, :].flatten(), self.dim_size)
            + xp.diag(diff_coeff_tensor[1, 0, :, :].flatten(), -self.dim_size)
        )
        # assemble matrix-version of diffusion operator
        L = D.T @ W @ D
        if is_numpy:
            L = sp.csr_matrix(L)
        elif is_cupy:
            # TODO: modify to allow GPU implementation
            ...
            # L = csp.csr_matrix(L)
        else:
            raise Warning(
                "Matrix-based sparse implementation only supports numpy or cupy diffusion coefficient. Assembling full matrices."
            )

        self._grad_matrix_based = pyca.LinOp.from_array(L)

    def asloss(self, data: pyct.NDArray = None) -> NotImplemented:
        """
        Notes
        -------
        DivergenceDiffusionOp class is not meant to be used to define a loss functional.
        """
        return NotImplemented

    def apply(self, arr: pyct.NDArray) -> NotImplemented:
        r"""
        Notes
        -------
        Divergence-based diffusion operators may arise from a variational formulation. This is true, e.g.,
        for the isotropic Perona-Malik, TV, Tikhonov. For these cases, it is possible
        to define the associated energy functional. If not implemented by the user in a daughter class, the method raises an error.
        """
        raise NotImplementedError

    def _compute_divergence_term(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
        z = self.gradient(arr)  # (batch,ndims,nchannels,nx,ny)
        if self.diffusion_coefficient is not None:
            # compute flux
            diffusion_coefficient = self.diffusion_coefficient(
                arr
            )  # (batch,ndims,nchannels,nx,ny) if iso, (batch,ndims,ndims,nchannels,nx,ny) if aniso
            # (1,2,120,40) or (batch,2,2,ndims,120,40) if iso/aniso
            # y_div = diffusion_coefficient(y_div)
            z = xp.einsum(self.einsum_subs, diffusion_coefficient, z)  # (batch,ndims,nchannels,nx,ny)
            # xp.einsum('...ijklm,...jklm->...iklm',diff_coeff,grad2) # to deal with arbitrary batch dimensions? we could do it vectorizing diffusion_coefficeint.apply at init
        # apply divergence
        y_div = self.gradient.T(z)  # (batch,nchannels,nx,ny)
        if self.outer_diffusivity is not None:
            outer_diffusivity = self.outer_diffusivity(arr)  # (batch,nchannels,nx,ny)
            # rescale divergence
            y_div *= outer_diffusivity
        return y_div  # (batch,nchannels,nx,ny)

    def _compute_extra_diffusion_term(self, arr: pyct.NDArray) -> pyct.NDArray:
        extra_term = self.extra_diffusion_term(arr)  # (batch,nchannels,nx,ny)
        return -extra_term

    def _compute_trace_term(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
        hessian = self.hessian(
            arr
        )  # (batch,nchannels,3,nx,ny)  since hessian currently does not allow mulit-d, nchannels as batching dimension
        hessian = xp.moveaxis(hessian, -3, -1)  # (batch,nchannels,nx,ny,3)
        hessian_full = hessian[..., self.full_matrix_indices]  # (batch,nchannels,nx,ny,4)
        hessian_full = hessian_full.reshape(
            *hessian_full.shape[:-1], self.ndims, self.ndims
        )  # (batch,nchannels,nx,ny,2,2)
        trace_tensor = self.trace_diffusion_coefficient(arr)  # (batch,2,2,nchannels,nx,ny)
        trace_tensor = xp.moveaxis(trace_tensor, [-5, -4], [-2, -1])  # (batch,nchannels,nx,ny,2,2)
        t_onto_h = xp.einsum("...ij,...jk", trace_tensor, hessian_full)  # (batch,nchannels,nx,ny,2,2)
        y_trace = t_onto_h[..., 0, 0] + t_onto_h[..., 1, 1]  # (batch,nchannels,nx,ny)
        if self.outer_trace_diffusivity is not None:
            outer_trace_diffusivity = self.outer_trace_diffusivity(arr)  # (batch,nchannels,nx,ny)
            # rescale trace
            y_trace *= outer_trace_diffusivity(arr)
        return -y_trace  # (batch,nchannels,nx,ny)

        # # hessian = self.hessian.unravel(self.hessian(arr)).squeeze().reshape(1, -1)
        # hessian = self.hessian.unravel(self.hessian(arr)).reshape(self.nchannels, -1)
        # trace_tensor = self.trace_diffusion_coefficient(arr)
        # y_trace = trace_tensor(hessian)
        # if self.outer_trace_diffusivity is not None:
        #     outer_trace_diffusivity = self.outer_trace_diffusivity(arr)
        #     # rescale trace
        #     y_trace *= outer_trace_diffusivity(arr)
        # return -y_trace

    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
        if self.matrix_based_impl:
            arr = arr.reshape(*arr.shape[:-2], self.num_pixels)
            y = self._grad_matrix_based(arr)
            y = y.reshape(*y.shape[:-1], *self.dim_shape[1:])
            return y
        else:
            y = xp.zeros_like(arr)
            if self.diffusion_coefficient is not None:
                # compute divergence term
                y += self._compute_divergence_term(arr)
            if self.extra_diffusion_term is not None:
                # compute extra diffusion term term
                y += self._compute_extra_diffusion_term(arr)
            if self.trace_diffusion_coefficient is not None:
                # compute trace tensor term
                y += self._compute_trace_term(arr)
            return y


# class DivergenceDiffusionOp(_DiffusionOp):
#     r"""
#     Class for divergence-based diffusion operators [see `Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_].
#
#     This class provides an interface to deal with divergence-based diffusion operators in the context of PDE-based regularisation.
#     In particular, we consider diffusion processes that can be written as
#
#     .. math::
#         \frac{\partial\mathbf{f}}{\partial t} = \mathbf{D}_{out}\mathrm{div}(\mathbf{D}_{in}\nabla\mathbf{f}),
#
#     where
#         * :math:`\mathbf{D}_{out} = \mathbf{D}_{out}(\mathbf{f})` is the outer diffusivity for the divergence term;
#         * :math:`\mathbf{D}_{in} = \mathbf{D}_{in}(\mathbf{f})` is the diffusion coefficient for the divergence term.
#
#     The right-hand side of the above PDE represents the output of the ``grad()`` method applied to the image :math:`\mathbf{f}`.
#
#     The effect of the :py:class:`~pyxu.operator.diffusion.DivergenceDiffusionOp` on an image :math:`\mathbf{f}` can be better understood
#     by focusing on a single pixel :math:`f_i` of the vectorisation of :math:`\mathbf{f}` (see, e.g., the discussion in
#     :py:class:`~pyxu.operator.diffusion._DiffusionCoefficient`).
#
#     Example
#     -------
#
#     .. plot::
#
#         import numpy as np
#         import matplotlib.pyplot as plt
#         import pyxu.operator.linop.diff as pydiff
#         import pyxu.operator.linop.filter as pyfilt
#         import pyxu.opt.solver as pysol
#         import pyxu.abc.solver as pysolver
#         import pyxu.opt.stop as pystop
#         import pyxu.operator.diffusion as pydiffusion
#         import skimage as skim
#         # Import image and corrupt it by noise
#         image = skim.color.rgb2gray(skim.data.cat())
#         print(image.shape) #(300, 451)
#         print(image.size) #135300
#         noise_std = 0.3*np.mean(image)
#         noisy_image = image + noise_std*np.random.randn(300, 451)
#         # Instantiate differential operators
#         # Gradient
#         grad = pydiff.Gradient(dim_shape=image.shape, diff_method="fd",
#                                                  mode="symmetric", scheme="forward")
#         # Gaussian gradient
#         gauss_grad = pydiff.Gradient(dim_shape=image.shape, diff_method="gd",
#                                                  mode="symmetric", sigma=2)
#
#         # Instantiate structure tensor (smooth_sigma=0 since we will consider edge enhancement)
#         structure_tensor = pyfilt.StructureTensor(dim_shape=image.shape, diff_method="gd", smooth_sigma=0,
#                                                   mode="symmetric", sigma=2)
#         # Estimate contrast parameter beta (heuristic, quantile-based)
#         gauss_grad_image=gauss_grad.unravel(gauss_grad(noisy_image.reshape(1,-1))).squeeze()
#         gauss_grad_norm = np.linalg.norm(gauss_grad_image, axis=0)
#         beta = np.quantile(gauss_grad_norm, 0.9)
#         # Define diffusion coefficients. For example, Perona-Malik (isotropic) and structure-tensor-based EdgeEnhancing (anisotropic)
#         # Perona-Malik coeff
#         PeronaMalik_diffusivity = pydiffusion.PeronaMalikDiffusivity(dim_shape=image.shape, gradient=gauss_grad, beta=beta, pm_fct="exponential")
#         PeronaMalik_diffusion_coeff = pydiffusion.DiffusionCoeffIsotropic(dim_shape=image.shape, diffusivity=PeronaMalik_diffusivity)
#
#         # Edge-enhancing coeff
#         EdgeEnhancing_diffusion_coeff = pydiffusion.DiffusionCoeffAnisoEdgeEnhancing(dim_shape=image.shape, structure_tensor=structure_tensor, beta=beta)
#         # Use defined diffusion coefficients to define two divergence-based diffusion operators
#         # Perona-Malik DiffusionOp
#         DivergenceDiffusionOpPM = pydiffusion.DivergenceDiffusionOp(dim_shape=image.shape, gradient=grad,
#                                                                   diffusion_coefficient=PeronaMalik_diffusion_coeff)
#         # Anisotropic Edge Enhancing DiffusionOp
#         DivergenceDiffusionOpEdge = pydiffusion.DivergenceDiffusionOp(dim_shape=image.shape, gradient=grad,
#                                                                   diffusion_coefficient=EdgeEnhancing_diffusion_coeff)
#         # Define stopping criterion and starting point
#         stop_crit = pystop.MaxIter(n=25)
#         x0 = noisy_image.reshape(1,-1)
#         # Perform 25 gradient flow iterations
#         PGD_PM = pysol.PGD(f = DivergenceDiffusionOpPM, g = None, show_progress=True, verbosity=100)
#         PGD_PM.fit(**dict(mode=pysolver.Mode.BLOCK, x0=x0, stop_crit=stop_crit, acceleration = False))
#         opt_PM = PGD_PM.solution()
#         PGD_Edge = pysol.PGD(f = DivergenceDiffusionOpEdge, g = None, show_progress=True, verbosity=100)
#         PGD_Edge.fit(**dict(mode=pysolver.Mode.BLOCK, x0=x0, stop_crit=stop_crit, acceleration = False))
#         opt_Edge = PGD_Edge.solution()
#         # Plot
#         fig, ax = plt.subplots(2,2,figsize=(12,9))
#         ax[0,0].imshow(image, cmap="gray")
#         ax[0,0].set_title("Image")
#         ax[0,0].axis('off')
#         ax[0,1].imshow(noisy_image, cmap="gray")
#         ax[0,1].set_title("Noisy image")
#         ax[0,1].axis('off')
#         ax[1,0].imshow(opt_PM.reshape(image.shape), cmap="gray")
#         ax[1,0].set_title("25 iterations Perona-Malik")
#         ax[1,0].axis('off')
#         ax[1,1].imshow(opt_Edge.reshape(image.shape), cmap="gray")
#         ax[1,1].set_title("25 iterations Anisotropic-Edge-Enhancing")
#         ax[1,1].axis('off')
#
#     """
#
#     def __init__(
#         self,
#         dim_shape: pyct.NDArrayShape,
#         nchannels: pyct.Integer = 1,
#         gradient: pyct.OpT = None,
#         outer_diffusivity: pyct.OpT = None,
#         diffusion_coefficient: pyct.OpT = None
#     ):
#         r"""
#
#         Parameters
#         ----------
#         dim_shape: pyct.Shape
#             Shape of the input array.
#         gradient: :py:class:`~pyxu.operator.linop.diff.Gradient`
#             Gradient operator. Defaults to `None`.
#         outer_diffusivity: :py:class:`~pyxu.operator.diffusion._Diffusivity`
#             Outer diffusivity operator, to be applied to the divergence term.
#         diffusion_coefficient: :py:class:`~pyxu.operator.diffusion._DiffusionCoefficient`
#             Diffusion coefficient operator of the divergence term.
#         prox_sigma: pyct.Real
#             Size of the structures that should be smoothed out by prox computation (when implemented in PnP-fashion).
#
#         Notes
#         ----
#         The parameter ``prox_sigma`` is expressed in pixel units. Independently of the ``sampling`` of the differential
#         operators involved, ``prox_sigma`` equal to `1` is meant to yield a prox which smoothes structures of the order
#         of one pixel.
#         """
#         super().__init__(
#             dim_shape=dim_shape,
#             nchannels=nchannels,
#             gradient=gradient,
#             outer_diffusivity=outer_diffusivity,
#             diffusion_coefficient=diffusion_coefficient
#         )
#         # estimate diff_lipschitz
#         _known_diff_lipschitz = False
#         if diffusion_coefficient is not None:
#             if diffusion_coefficient.bounded:
#                 _known_diff_lipschitz = True
#                 if not diffusion_coefficient.isotropic:
#                     # extra factor 2 in this case for exact expression?
#                     msg = "For anisotropic `diffusion_coefficient`, the estimated `diff_lipschitz` experimentally grants stability but is not guaranteed to hold."
#                     warnings.warn(msg)
#             if outer_diffusivity is not None:
#                 _known_diff_lipschitz = _known_diff_lipschitz and outer_diffusivity.bounded
#         if _known_diff_lipschitz:
#             self.diff_lipschitz = self.gradient.lipschitz ** 2
#
#     #@pycrt.enforce_precision(i="arr")
#     @pycu.vectorize("arr")
#     def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
#         arr = arr.reshape(self.nchannels, -1)
#         # arr = arr.reshape(1, -1)
#         # compute divergence term
#         y = self._compute_divergence_term(arr)
#         return y.reshape(1, -1)
#
#
# class SnakeDiffusionOp(_DiffusionOp):
#     r"""
#     Class for snake diffusion operators (active contour models) [see `Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_].
#
#     This class provides an interface to deal with snake diffusion operators in the context of PDE-based regularisation.
#     In particular, we consider diffusion processes that can be written as
#
#     .. math::
#         \frac{\partial\mathbf{f}}{\partial t} = \mathbf{D}_{out}\mathrm{div}(\mathbf{D}_{in}\nabla\mathbf{f})+ \mathbf{b},
#     where
#         * :math:`\mathbf{D}_{out} = \mathbf{D}_{out}(\mathbf{f})` is the outer diffusivity for the divergence term;
#         * :math:`\mathbf{D}_{in} = \mathbf{D}_{in}(\mathbf{f})` is the diffusion coefficient for the divergence term;
#         * :math:`\mathbf{b} = \mathbf{b}(\mathbf{f})` is the balloon force.
#
#     The right-hand side of the above PDE represents the output of the ``grad()`` method applied to the image :math:`\mathbf{f}`.
#
#     The effect of the :py:class:`~pyxu.operator.diffusion.SnakeDiffusionOp` on an image :math:`\mathbf{f}` can be better understood
#     by focusing on a single pixel :math:`f_i` of the vectorisation of :math:`\mathbf{f}` (see, e.g., discussion in
#     :py:class:`~pyxu.operator.diffusion._DiffusionCoefficient`).
#     """
#
#     def __init__(
#         self,
#         dim_shape: pyct.NDArrayShape,
#         gradient: pyct.OpT = None,
#         outer_diffusivity: pyct.OpT = None,
#         diffusion_coefficient: pyct.OpT = None,
#         balloon_force: pyct.OpT = None,
#         prox_sigma: pyct.Real = 2,
#     ):
#         r"""
#
#         Parameters
#         ----------
#         dim_shape: pyct.Shape
#             Shape of the input array.
#         gradient: :py:class:`~pyxu.operator.linop.diff.Gradient`
#             Gradient operator. Defaults to `None`.
#         outer_diffusivity: :py:class:`~pyxu.operator.diffusion._Diffusivity`
#             Outer diffusivity operator, to be applied to the divergence term.
#         diffusion_coefficient: :py:class:`~pyxu.operator.diffusion._DiffusionCoefficient`
#             Diffusion coefficient operator of the divergence term.
#         balloon_force: :py:class:`~pyxu.operator.diffusion._BalloonForce`
#             Balloon force operator.
#         prox_sigma: pyct.Real
#             Size of the structures that should be smoothed out by prox computation (when implemented in PnP-fashion).
#
#         Notes
#         ----
#         The parameter ``prox_sigma`` is expressed in pixel units. Independently of the ``sampling`` of the differential
#         operators involved, ``prox_sigma`` equal to `1` is meant to yield a prox which smoothes structures of the order
#         of one pixel.
#         """
#         super().__init__(
#             dim_shape=dim_shape,
#             gradient=gradient,
#             outer_diffusivity=outer_diffusivity,
#             diffusion_coefficient=diffusion_coefficient,
#             balloon_force=balloon_force,
#             prox_sigma=prox_sigma,
#         )
#         # estimate diff_lipschitz
#         _known_diff_lipschitz = False
#         if diffusion_coefficient is not None:
#             if diffusion_coefficient.bounded:
#                 _known_diff_lipschitz = True
#                 if not diffusion_coefficient.isotropic:
#                     # extra factor 2 in this case for exact expression?
#                     msg = "For anisotropic `diffusion_coefficient`, the estimated `diff_lipschitz` experimentally grants stability but is not guaranteed to hold."
#                     warnings.warn(msg)
#             if outer_diffusivity is not None:
#                 _known_diff_lipschitz = _known_diff_lipschitz and outer_diffusivity.bounded
#         if _known_diff_lipschitz:
#             self._diff_lipschitz = gradient.lipschitz() ** 2
#         if balloon_force is not None:
#             self._diff_lipschitz += balloon_force._lipschitz
#
#     #@pycrt.enforce_precision(i="arr")
#     @pycu.vectorize("arr")
#     def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
#         xp = pycu.get_array_module(arr)
#         arr = arr.reshape(1, -1)
#         y = xp.zeros_like(arr)
#         # compute divergence term
#         y += self._compute_divergence_term(arr)
#         # compute balloon force term
#         y += self._compute_balloon_term(arr)
#         return y
#
#
# class TraceDiffusionOp(_DiffusionOp):
#     r"""
#     Class for trace-based diffusion operators [see `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_].
#
#     This class provides an interface to deal with trace-based diffusion operators in the context of PDE-based regularisation.
#     In particular, we consider diffusion processes that can be written as
#
#     .. math::
#         \frac{\partial\mathbf{f}}{\partial t} = \mathbf{T}_{out}\mathrm{trace}\big(\mathbf{T}_{in}\mathbf{H}(\mathbf{f})\big),
#
#     where
#         * :math:`\mathbf{T}_{out} = \mathbf{T}_{out}(\mathbf{f})` is the outer diffusivity for the trace term;
#         * :math:`\mathbf{T}_{in} = \mathbf{T}_{in}(\mathbf{f})` is the diffusion coefficient for the trace term.
#
#     The right-hand side of the above PDE represents the output of the ``grad()`` method applied to the image :math:`\mathbf{f}`.
#
#     The effect of the :py:class:`~pyxu.operator.diffusion.TraceDiffusionOp` on an image :math:`\mathbf{f}` can be better understood
#     by focusing on a single pixel :math:`f_i` of the vectorisation of :math:`\mathbf{f}` (see, e.g., discussion in
#     :py:class:`~pyxu.operator.diffusion._DiffusionCoefficient`).
#
#     Example
#     -------
#
#     .. plot::
#
#         import numpy as np
#         import matplotlib.pyplot as plt
#         import pyxu.operator.linop.diff as pydiff
#         import pyxu.operator.linop.filter as pyfilt
#         import pyxu.opt.solver as pysol
#         import pyxu.abc.solver as pysolver
#         import pyxu.opt.stop as pystop
#         import pyxu.operator.diffusion as pydiffusion
#         import skimage as skim
#         # Import image
#         image = skim.color.rgb2gray(skim.data.cat())
#         print(image.shape) #(300, 451)
#         print(image.size) #135300
#         noise_std = 0.3*np.mean(image)
#         noisy_image = image + noise_std*np.random.randn(300, 451)
#         # Instantiate needed differential operators
#         # Gaussian gradient operator
#         gauss_grad = pydiff.Gradient(dim_shape=image.shape, diff_method="gd",
#                                                  mode="symmetric", sigma=2)
#         # Hessian operator
#         hessian = pydiff.Hessian(dim_shape=image.shape, diff_method="fd", mode="symmetric",
#                                                    scheme="central", accuracy=2)
#         # Instantiate structure tensor
#         structure_tensor = pyfilt.StructureTensor(dim_shape=image.shape, diff_method="gd", smooth_sigma=0,
#                                                   mode="symmetric", sigma=2)
#         # Estimate contrast parameter beta (heuristic, quantile-based)
#         gauss_grad_image=gauss_grad.unravel(gauss_grad(noisy_image.reshape(1,-1))).squeeze()
#         gauss_grad_norm = np.linalg.norm(gauss_grad_image, axis=0)
#         beta = np.quantile(gauss_grad_norm, 0.9)
#         # Define two different diffusion coefficients: Perona-Malik (isotropic) and structure-tensor-based EdgeEnhancing (anisotropic)
#         # Perona-Malik coeff
#         PeronaMalik_diffusivity = pydiffusion.PeronaMalikDiffusivity(dim_shape=image.shape, gradient=gauss_grad, beta=beta, pm_fct="exponential")
#         PeronaMalik_diffusion_coeff = pydiffusion.DiffusionCoeffIsotropic(dim_shape=image.shape, diffusivity=PeronaMalik_diffusivity,
#                                                                           trace_term=True)
#         # Edge-enhancing coeff
#         EdgeEnhancing_diffusion_coeff = pydiffusion.DiffusionCoeffAnisoEdgeEnhancing(dim_shape=image.shape,
#                                                                                      structure_tensor=structure_tensor,
#                                                                                      beta=beta, trace_term=True)
#         # Use defined diffusion coefficients to define two trace-based diffusion operators
#         # Perona-Malik DiffusionOp
#         TraceDiffusionOpPM = pydiffusion.TraceDiffusionOp(dim_shape=image.shape, hessian=hessian,
#                                                                   trace_diffusion_coefficient=PeronaMalik_diffusion_coeff)
#         # Anisotropic Edge Enhancing DiffusionOp
#         TraceDiffusionOpEdge = pydiffusion.TraceDiffusionOp(dim_shape=image.shape, hessian=hessian,
#                                                                   trace_diffusion_coefficient=EdgeEnhancing_diffusion_coeff)
#         # Define stopping criterion and starting point
#         stop_crit = pystop.MaxIter(n=25)
#         x0 = noisy_image.reshape(1,-1)
#         # Perform 25 gradient flow iterations
#         PGD_PM = pysol.PGD(f = TraceDiffusionOpPM, g = None, show_progress=True, verbosity=100)
#         PGD_PM.fit(**dict(mode=pysolver.Mode.BLOCK, x0=x0, stop_crit=stop_crit, acceleration = False))
#         opt_PM = PGD_PM.solution()
#         PGD_Edge = pysol.PGD(f = TraceDiffusionOpEdge, g = None, show_progress=True, verbosity=100)
#         PGD_Edge.fit(**dict(mode=pysolver.Mode.BLOCK, x0=x0, stop_crit=stop_crit, acceleration = False))
#         opt_Edge = PGD_Edge.solution()
#         # Plot
#         fig, ax = plt.subplots(2,2,figsize=(12,9))
#         ax[0,0].imshow(image, cmap="gray")
#         ax[0,0].set_title("Image")
#         ax[0,0].axis('off')
#         ax[0,1].imshow(noisy_image, cmap="gray")
#         ax[0,1].set_title("Noisy image")
#         ax[0,1].axis('off')
#         ax[1,0].imshow(opt_PM.reshape(image.shape), cmap="gray")
#         ax[1,0].set_title("25 iterations Perona-Malik")
#         ax[1,0].axis('off')
#         ax[1,1].imshow(opt_Edge.reshape(image.shape), cmap="gray")
#         ax[1,1].set_title("25 iterations Anisotropic-Edge-Enhancing")
#         ax[1,1].axis('off')
#
#     """
#
#     def __init__(
#         self,
#         dim_shape: pyct.NDArrayShape,
#         nchannels: pyct.Integer = 1,
#         hessian: pyct.OpT = None,
#         outer_trace_diffusivity: pyct.OpT = None,
#         trace_diffusion_coefficient: pyct.OpT = None
#     ):
#         r"""
#
#         Parameters
#         ----------
#         dim_shape: pyct.Shape
#             Shape of the pixelised image.
#         hessian:  :py:class:`~pyxu.operator.linop.diff.Hessian`
#             Hessian operator. Defaults to `None`.
#         outer_trace_diffusivity: :py:class:`~pyxu.operator.diffusion._Diffusivity`
#             Outer diffusivity operator, to be applied to the trace term.
#         trace_diffusion_coefficient: :py:class:`~pyxu.operator.diffusion._DiffusionCoefficient`
#             Diffusion coefficient operator of the trace term.
#         prox_sigma: pyct.Real
#             Size of the structures that should be smoothed out by prox computation (when implemented in PnP-fashion).
#
#         Notes
#         ----
#         The parameter ``prox_sigma`` is expressed in pixel units. Independently of the ``sampling`` of the differential
#         operators involved, ``prox_sigma`` equal to `1` is meant to yield a prox which smoothes structures of the order
#         of one pixel.
#         """
#         super().__init__(
#             dim_shape=dim_shape,
#             nchannels=nchannels,
#             outer_trace_diffusivity=outer_trace_diffusivity,
#             trace_diffusion_coefficient=trace_diffusion_coefficient,
#             hessian=hessian
#         )
#         # estimate diff_lipschitz (further think, extra factors may arise for trace case)
#         _known_diff_lipschitz = False
#         if trace_diffusion_coefficient is not None:
#             if trace_diffusion_coefficient.bounded:
#                 _known_diff_lipschitz = True
#                 if not trace_diffusion_coefficient.isotropic:
#                     msg = "For anisotropic `trace_diffusion_coefficient`, the estimated `diff_lipschitz` experimentally grants stability but is not guaranteed to hold."
#                     warnings.warn(msg)
#             if outer_trace_diffusivity is not None:
#                 _known_diff_lipschitz = _known_diff_lipschitz and outer_trace_diffusivity.bounded
#         if _known_diff_lipschitz:
#             self.diff_lipschitz = hessian.lipschitz
#
#     #@pycrt.enforce_precision(i="arr")
#     @pycu.vectorize("arr")
#     def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
#         xp = pycu.get_array_module(arr)
#         arr = arr.reshape(self.nchannels, -1)
#         # arr = arr.reshape(1, -1)
#         y = xp.zeros_like(arr)
#         # compute trace tensor term
#         y += self._compute_trace_term(arr)
#         return y.reshape(1, -1)
#
#
# class CurvaturePreservingDiffusionOp(_DiffusionOp):
#     r"""
#     Class for curvature preserving diffusion operators [see `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_].
#
#     This class provides an interface to deal with curvature preserving diffusion operators in the context of PDE-based regularisation.
#     In particular, we consider diffusion processes that can be written as
#
#     .. math::
#         \frac{\partial\mathbf{f}}{\partial t} = \mathbf{T}_{out}\mathrm{trace}\big(\mathbf{T}_{in}\mathbf{H}(\mathbf{f})\big) + (\nabla\mathbf{f})^T \mathbf{J}_{\mathbf{w}}\mathbf{w},
#
#     where
#         * :math:`\mathbf{T}_{out} = \mathbf{T}_{out}(\mathbf{f})` is the outer diffusivity for the trace term;
#         * :math:`\mathbf{T}_{in} = \mathbf{T}_{in}(\mathbf{f})` is the diffusion coefficient for the trace term;
#         * :math:`\mathbf{w}` is a vector field assigning a :math:`2`-dimensional vector to each pixel;
#         * :math:`\mathbf{J}_\mathbf{w}` is the Jacobian of the vector field :math:`\mathbf{w}`.
#
#     The right-hand side of the above PDE represents the output of the ``grad()`` method applied to the image :math:`\mathbf{f}`.
#
#     The resulting smoothing process tries to preserve the curvature of the vector field :math:`\mathbf{w}`.
#
#     The effect of the :py:class:`~pyxu.operator.diffusion.CurvaturePreservingDiffusionOp` on an image :math:`\mathbf{f}` can be better understood
#     by focusing on a single pixel :math:`f_i` of the vectorisation of :math:`\mathbf{f}` (see, e.g., discussion in
#     :py:class:`~pyxu.operator.diffusion._DiffusionCoefficient`).
#
#     Example
#     -------
#
#     .. plot::
#
#         import numpy as np
#         import matplotlib.pyplot as plt
#         import pyxu.operator.linop.diff as pydiff
#         import pyxu.opt.solver as pysol
#         import pyxu.abc.solver as pysolver
#         import pyxu.opt.stop as pystop
#         import pyxu.operator.diffusion as pydiffusion
#         import skimage as skim
#         # Import image
#         image = skim.color.rgb2gray(skim.data.cat())
#         # Instantiate needed differential operators
#         # Gradient operator
#         grad = pydiff.Gradient(dim_shape=image.shape, diff_method="fd",
#                                                  mode="symmetric", scheme="forward")
#         # Gaussian gradient operator
#         gauss_grad = pydiff.Gradient(dim_shape=image.shape, diff_method="gd",
#                                                  mode="symmetric", sigma=2)
#         # Hessian operator
#         hessian = pydiff.Hessian(dim_shape=image.shape, diff_method="fd", mode="symmetric",
#                                                    scheme="central", accuracy=2)
#         # Define vector field, diffusion process will preserve curvature along it
#         image_center=np.array(image.shape)/2+[0.25, 0.25]
#         curvature_preservation_field=np.zeros((2,image.size))
#         curv_pres_1 = np.zeros(image.shape)
#         curv_pres_2 = np.zeros(image.shape)
#         for i in range(image.shape[0]):
#             for j in range(image.shape[1]):
#                 theta = np.arctan2(-i+image_center[0], j-image_center[1])
#                 curv_pres_1[i,j] = np.cos(theta)
#                 curv_pres_2[i,j] = np.sin(theta)
#         curvature_preservation_field[0,:]=curv_pres_1.reshape(1,-1)
#         curvature_preservation_field[1,:]=curv_pres_2.reshape(1,-1)
#         # Define curvature-preserving diffusion operator
#         CurvPresDiffusionOpPM = pydiffusion.CurvaturePreservingDiffusionOp(dim_shape=image.shape,
#                                                                            gradient=grad, hessian=hessian,
#                                                                   curvature_preservation_field=curvature_preservation_field)
#         # Define stopping criterion and starting point
#         stop_crit = pystop.MaxIter(n=500)
#         x0 = image.reshape(1,-1)
#         # Perform 500 gradient flow iterations
#         PGD_curve = pysol.PGD(f = CurvPresDiffusionOpPM, g = None, show_progress=True, verbosity=100)
#         PGD_curve.fit(**dict(mode=pysolver.Mode.BLOCK, x0=x0, stop_crit=stop_crit, acceleration = False))
#         opt_curve = PGD_curve.solution()
#         # Plot
#         fig, ax = plt.subplots(1,3,figsize=(20,4))
#         ax[0].imshow(image, cmap="gray", aspect="auto")
#         ax[0].set_title("Image")
#         ax[0].axis('off')
#         ax[1].quiver(curv_pres_2[::40,::60], curv_pres_1[::40,::60])
#         ax[1].set_title("Vector field")
#         ax[1].axis('off')
#         ax[2].imshow(opt_curve.reshape(image.shape), cmap="gray", aspect="auto")
#         ax[2].set_title("500 iterations Curvature Preserving")
#         ax[2].axis('off')
#
#     """
#
#     def __init__(
#         self,
#         dim_shape: pyct.NDArrayShape,
#         gradient: pyct.OpT = None,
#         hessian: pyct.OpT = None,
#         curvature_preservation_field: pyct.NDArray = None,
#         prox_sigma: pyct.Real = 2,
#     ):
#         r"""
#
#         Parameters
#         ----------
#         dim_shape: pyct.Shape
#             Shape of the pixelised image.
#         gradient: :py:class:`~pyxu.operator.linop.diff.Gradient`
#             Gradient operator. Defaults to `None`.
#         hessian:  :py:class:`~pyxu.operator.linop.diff.Hessian`
#             Hessian operator. Defaults to `None`.
#         curvature_preservation_field: pyct.NDArray
#             Vector field along which curvature should be preserved. Defaults to `None`.
#         prox_sigma: pyct.Real
#             Size of the structures that should be smoothed out by prox computation (when implemented in PnP-fashion).
#
#         Notes
#         ----
#         The parameter ``prox_sigma`` is expressed in pixel units. Independently of the ``sampling`` of the differential
#         operators involved, ``prox_sigma`` equal to `1` is meant to yield a prox which smoothes structures of the order
#         of one pixel.
#         """
#         if hessian is None:
#             msg = "\n".join(
#                 [
#                     "No `hessian` was passed, needed for trace term involving `trace_diffusion_coefficient`.",
#                     "Initializing a central finite difference operator with unitary sampling as default.",
#                 ]
#             )
#             warnings.warn(msg)
#             hessian = pydiff.Hessian(
#                 dim_shape=dim_shape,
#                 diff_method="fd",
#                 mode="symmetric",
#                 sampling=1.0,
#                 scheme="central",
#                 accuracy=2,
#             )
#         super().__init__(
#             dim_shape=dim_shape,
#             curvature_preservation_field=curvature_preservation_field,
#             gradient=gradient,
#             hessian=hessian,
#             prox_sigma=prox_sigma,
#         )
#         # assemble trace diffusion coefficient corresponding to the curvature preserving field
#         curvature_preservation_field = curvature_preservation_field.T
#         tensors = curvature_preservation_field.reshape(self.dim, self.ndims, 1) * curvature_preservation_field.reshape(
#             self.dim, 1, self.ndims
#         )
#         trace_diffusion_coefficient = tensors.reshape(self.dim, -1)
#         ops = []
#         for i in range(self.ndims):
#             # only upper diagonal entries are considered (symmetric tensors)
#             first_idx = i
#             for j in np.arange(first_idx, self.ndims):
#                 op = pybase.DiagonalOp(trace_diffusion_coefficient[:, i * self.ndims + j])
#                 if j > i:
#                     # multiply by 2 extra diagonal elements
#                     op *= 2.0
#                 ops.append(op)
#         self.trace_diffusion_coefficient = diffcoeff._DiffusionCoefficient(dim_shape=dim_shape, isotropic=False, trace_term=True)
#         self.trace_diffusion_coefficient.set_frozen_op(pyblock.hstack(ops))
#         # estimate diff_lipschitz
#         self._diff_lipschitz = hessian.lipschitz()
#         if self.curvature_preservation_field is not None:
#             max_norm = np.max(np.linalg.norm(curvature_preservation_field, axis=1))
#             self._diff_lipschitz *= max_norm
#             # abs(<gradient(u), J_w(w)>) \leq norm(gradient(u)) * norm(J_w(w))
#             # \leq L_grad*norm(u)*2*L_grad*(norm(w)**2) = 2*L_grad**2 * norm(u)
#             self._diff_lipschitz += 2 * (gradient.lipschitz() ** 2) * max_norm
#
#     #@pycrt.enforce_precision(i="arr")
#     @pycu.vectorize("arr")
#     def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
#         xp = pycu.get_array_module(arr)
#         arr = arr.reshape(1, -1)
#         y = xp.zeros_like(arr)
#         # compute trace tensor term
#         y += self._compute_trace_term(arr)
#         # compute curvature preserving term
#         y += self._compute_curvature_preserving_term(arr)
#         return y
#
#
# class TV_DiffusionOp(pyca.DiffFunc):
#     # TV diffusion operator featuring second order in space discretization scheme for gradient
#     def asloss(self, data: pyct.NDArray = None) -> pyct.OpT:
#         pass
#
#     def __init__(self, dim_shape: pyct.NDArrayShape, beta=1e-3):
#         self.dim_shape = dim_shape
#         self.ndims = len(self.dim_shape)
#         super().__init__(shape=(1, int(np.prod(dim_shape))))
#         filter_ = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
#         self.S = pycop.Stencil(kernel=filter_, center=(1, 1), dim_shape=dim_shape)
#         self.second_derivative = pydiff.Hessian(
#             dim_shape=dim_shape, directions=[(0, 0), (1, 1)], accuracy=2, scheme="central", mode="symmetric"
#         )
#         self.central_grad = pydiff.Gradient(dim_shape=self.dim_shape, scheme="central", mode="symmetric")
#         self.beta = beta
#         self._diff_lipschitz = 4
#
#     def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
#         # Inplace implementation of
#         #   beta*sum(xp.sqrt(beta**2+grad_norm_sq))
#         xp = pycu.get_array_module(arr)
#         grad_arr = self.grad.unravel(self.grad(arr))
#         grad_arr **= 2
#         y = xp.sum(grad_arr, axis=1, keepdims=False)
#         y += self.beta**2
#         y = xp.sqrt(y)
#         z = xp.sum(y, axis=-1)
#         return self.beta * z
#
#     def grad(self, arr):
#         # gradient implementation obtained from proper second-order in space finite difference discretization
#         grad_z_central = self.central_grad.unravel(self.central_grad(arr)).squeeze().reshape(2, -1)
#         grad_z_central_norm_sq = np.sum(grad_z_central**2, axis=0, keepdims=False)
#         norm_term = (grad_z_central_norm_sq + self.beta**2) ** (-3 / 2)
#         second_deriv = self.second_derivative.unravel(self.second_derivative(arr)).squeeze().reshape(2, -1)
#         A = second_deriv[0, :] * (grad_z_central[1, :] ** 2 + self.beta**2)
#         B = second_deriv[1, :] * (grad_z_central[0, :] ** 2 + self.beta**2)
#         C = (-0.5 * grad_z_central[0, :] * grad_z_central[1, :] * self.S(arr).reshape(1, -1)).flatten()
#         return -self.beta * norm_term * (A + B + C)
#
#
# class DivergenceDiffusionOp_NewDiscr(_DiffusionOp):
#     def __init__(
#         self,
#         dim_shape: pyct.NDArrayShape,
#         nchannels: pyct.Integer = 1,
#         gradient: pyct.OpT = None,
#         outer_diffusivity: pyct.OpT = None,
#         diffusion_coefficient: pyct.OpT = None,
#     ):
#         super().__init__(
#             dim_shape=dim_shape,
#             nchannels=nchannels,
#             gradient=gradient,
#             outer_diffusivity=outer_diffusivity,
#             diffusion_coefficient=diffusion_coefficient,
#         )
#         # estimate diff_lipschitz
#         # _known_diff_lipschitz = False
#         # if diffusion_coefficient:
#         #     if diffusion_coefficient.bounded:
#         #         _known_diff_lipschitz = True
#         #         if not diffusion_coefficient.isotropic:
#         #             # extra factor 2 in this case for exact expression?
#         #             msg = "For anisotropic `diffusion_coefficient`, the estimated `diff_lipschitz` experimentally grants stability but is not guaranteed to hold."
#         #             warnings.warn(msg)
#         #     if outer_diffusivity:
#         #         _known_diff_lipschitz = _known_diff_lipschitz and outer_diffusivity.bounded
#         # if _known_diff_lipschitz:
#         #     self._diff_lipschitz = gradient.lipschitz() ** 2
#         self.diff_lipschitz = gradient.lipschitz ** 2
#
#         filter_extradiag_x = np.array([1 / (2 * self.gradient.meta.sampling[0][0]), 0, -1 / (2 * self.gradient.meta.sampling[0][0])])
#         self.S_extradiag_x = pycop.Stencil(
#             kernel=(filter_extradiag_x, np.array([1])), center=(1, 0), dim_shape=dim_shape, mode="symmetric"
#         )
#         filter_extradiag_y = np.array([-1 / (2 * self.gradient.meta.sampling[0][1]), 0, 1 / (2 * self.gradient.meta.sampling[0][1])])
#         self.S_extradiag_y = pycop.Stencil(
#             kernel=(np.array([1]), filter_extradiag_y), center=(0, 1), dim_shape=dim_shape, mode="symmetric"
#         )
#         # filters for approximation evaluating diffusivities at midpoints via shifting
#         filter_midpoint = np.array([0.5, 0.5])
#         self.S_midx = pycop.Stencil(
#             kernel=(filter_midpoint, np.array([1])), center=(0, 0), dim_shape=dim_shape, mode="symmetric"
#         )
#         self.S_midy = pycop.Stencil(
#             kernel=(np.array([1]), filter_midpoint), center=(0, 0), dim_shape=dim_shape, mode="symmetric"
#         )
#
#         self.frozen_diffcoeff = False
#
#     ##@pycrt.enforce_precision(i="arr")
#     #@pycu.vectorize("arr")
#     def freeze_diffusion_coefficient(self, arr: pyct.NDArray):
#         self.compute_diffusion_coefficient(arr)
#         self.frozen_diffcoeff = True
#         return
#
#     ##@pycrt.enforce_precision(i="arr")
#     #@pycu.vectorize("arr")
#     def compute_diffusion_coefficient(self, arr: pyct.NDArray):
#         if not self.frozen_diffcoeff:
#             u, e = self.diffusion_coefficient._eigendecompose_struct_tensor(arr)
#             lambdas = self.diffusion_coefficient._compute_intensities(e)
#             tensors = self.diffusion_coefficient._assemble_tensors(u, lambdas)
#
#             # for Weickert's C term (it's our A!)
#             shifted_coeffs_x = self.S_midx(tensors[:, 0, 0])
#             # for Weickert's A term (it's ours C!)
#             shifted_coeffs_y = self.S_midy(tensors[:, 1, 1])
#
#             self.eval_diffusion_coeff = np.hstack((shifted_coeffs_x, shifted_coeffs_y))
#
#             # for Weickert's B term
#             self.eval_B = tensors[:, 0, 1].reshape(1, -1)
#         return
#
#     #@pycrt.enforce_precision(i="arr")
#     @pycu.vectorize("arr")
#     def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
#         xp = pycu.get_array_module(arr)
#         arr = arr.reshape(self.nchannels, -1)
#
#         # compute diffusion coefficient (if not frozen)
#         self.compute_diffusion_coefficient(arr)
#
#         out = xp.zeros(arr.shape, dtype=arr.dtype)
#         # assemble C, A terms
#         grad_ = self.gradient(arr)
#         div_arg = self.eval_diffusion_coeff * grad_
#         out += self.gradient.T(div_arg)
#         ed1_ = self.S_extradiag_x(arr) * self.eval_B
#         ed1 = self.S_extradiag_y(ed1_)
#         ed2_ = self.S_extradiag_y(arr) * self.eval_B
#         ed2 = self.S_extradiag_x(ed2_)
#         out += ed1
#         out += ed2
#         return out.reshape(1, -1)
#
#
# class _DiffusionOpOld(pyca.ProxDiffFunc):
#     r"""
#     Abstract class for diffusion operators
#     [see `Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_ and `Tschumperle-Deriche <https://hal.science/hal-00332798/document>`_].
#
#     Notes
#     -----
#
#     This class provides an interface to deal with PDE-based regularisation. For simplicity, throughout the
#     documentation we consider a :math:`2`-dimensional signal :math:`\mathbf{f} \in \mathbb{R}^{N_{0} \times N_1}`,
#     but higher dimensional signals could be considered. We denote by :math:`f_i` the :math:`i`-th entry (pixel)
#     of the vectorisation of :math:`\mathbf{f}`, :math:`i=0,\dots,(N_0N_1-1)`. Furthermore, let
#     :math:`(\nabla \mathbf{f})_i = \big((\frac{\partial \mathbf{f}}{\partial x_0})_i,\dots,(\frac{\partial \mathbf{f}}{\partial x_{D-1}})_i\big)^T`.
#
#     To give some intuition, let us first consider a simple case: Tikhonov regularisation, corresponding to linear, isotropic,
#     homogeneous smoothing. This yields the regularisation functional
#
#     .. math::
#         \phi(\mathbf{f}) = \frac{1}{2}\Vert \nabla \mathbf{f} \Vert_2^2
#                          = \frac{1}{2}\sum_{i=0}^{N_0 N_1-1} \Vert (\nabla \mathbf{f})_i \Vert_2^2.
#
#     Then, we have
#
#     .. math::
#         \nabla \phi(\mathbf{f}) = \nabla^T\nabla\mathbf{f}
#                                 = -\mathrm{div}(\nabla\mathbf{f})
#
#     where :math:`\nabla^T` is the adjoint of the gradient operator and where we exploited the fact that
#     :math:`\nabla^T = -\mathrm{div}`, the divergence. To solve the optimization problem
#
#     .. math::
#         \underset{\mathbf{f}}{\mathrm{argmin}} \ \phi(\mathbf{f}),
#
#     we could apply gradient descent, whose update formula is given by
#
#     .. math::
#         \mathbf{f}_{n+1} = \mathbf{f}_n + \eta \mathrm{div}(\nabla\mathbf{f}_n),
#
#     where :math:`\mathbf{f}_n` is the :math:`n`-th iterate and :math:`\eta` is the step size of the algorithm. The above
#     update equation can be interpreted as one step in time of the explicit Euler integration method applied to the PDE
#
#     .. math::
#         \frac{\partial\mathbf{f}}{\partial t} = \mathrm{div}(\nabla\mathbf{f})
#                                               = \Delta \mathbf{f},
#
#     with the step size :math:`\Delta t=\eta` in time.
#     This time-dependent PDE represents the gradient flow formulation of the original optimization problem, where the
#     time is artificial and characterises the optimization process. We recognise, moreover, that we obtain the
#     well-known heat equation.
#
#     We can thus let the PDE evolve in time until it reaches a steady state :math:`\frac{\partial\mathbf{f}}{\partial t}=0`.
#     The solution will therefore satisfy the first order optimality condition :math:`\nabla \phi(\mathbf{f})=0`.
#
#     If formulated as above, a trivial steady state corresponding to an infinitely flat solution will be obtained.
#     However, if the functional :math:`\phi(\cdot)` is combined with a data-fidelity functional :math:`\ell(\cdot)`
#     in the context of an inverse problem, an extra term :math:`\nabla \ell(\cdot)` will arise in the gradient flow
#     formulation. This will lead to a non-trivial steady state corresponding to the balance between the data-fidelity
#     and regularisation terms.
#
#     In the context of PDE-based regularisation, it is not necessary to limit ourselves to consider cases where it is possible
#     to explicitly define a variational functional :math:`\phi(\cdot)`. In the spirit of Plug&Play (PnP) approaches,
#     we can consider diffusion operators that are only characterised by their smoothing action in gradient flow form: no
#     underlying functional :math:`\phi(\cdot)` may exist. This allows to study complex diffusion processes designed to
#     enhance specific features of the image.
#
#     In particular, we consider diffusion processes that, in their most general form, can be written as the composite term
#
#     .. math::
#         \frac{\partial\mathbf{f}}{\partial t} = \mathbf{D}_{out}\mathrm{div}(\mathbf{D}_{in}\nabla\mathbf{f})
#         + \mathbf{b} + \mathbf{T}_{out}\mathrm{trace}\big(\mathbf{T}_{in}\mathbf{H}(\mathbf{f})\big) + (\nabla\mathbf{f})^T \mathbf{J}_{\mathbf{w}}\mathbf{w},
#
#     where
#         * :math:`\mathbf{D}_{out} = \mathbf{D}_{out}(\mathbf{f}) \in \mathbb{R}^{N_{tot} \times N_{tot} }` is the outer diffusivity for the divergence term;
#         * :math:`\mathbf{D}_{in} = \mathbf{D}_{in}(\mathbf{f}) \in \mathbb{R}^{D N_{tot} \times D N_{tot} }` is the diffusion coefficient for the divergence term;
#         * :math:`\mathbf{b} = \mathbf{b}(\mathbf{f}) \in \mathbb{R}^{N_{tot}}` is the balloon force;
#         * :math:`\mathbf{T}_{out} = \mathbf{T}_{out}(\mathbf{f}) \in \mathbb{R}^{N_{tot} \times N_{tot} }` is the outer diffusivity for the trace term;
#         * :math:`\mathbf{T}_{in} = \mathbf{T}_{in}(\mathbf{f}) \in \mathbb{R}^{D^2 N_{tot} \times D^2 N_{tot} }` is the diffusion coefficient for the trace term;
#         * :math:`\mathbf{w} \in \mathbb{R}^{D N_{tot}}` is a vector field assigning a :math:`2`-dimensional vector to each pixel;
#         * :math:`\mathbf{J}_\mathbf{w} \in \mathbb{R}^{D N_{tot}^2 \times D N_{tot}}` is the Jacobian of the vector field :math:`\mathbf{w}`.
#
#     The right-hand side of the above PDE represents the output of the ``grad()`` method applied to the image :math:`\mathbf{f}`.
#
#     To conclude, we remark that the effect of the diffusion operator on an image :math:`\mathbf{f}` can be better understood
#     by focusing on a single pixel :math:`f_i` of the vectorisation of :math:`\mathbf{f}` (see, e.g., discussion in
#     :py:class:`~pyxu.operator.diffusion._DiffusionCoefficient`).
#
#     **Remark 1**
#
#     ``_DiffusionOp`` is an atypical :py:class:`~pyxu.abc.operator.ProxDiffFunc`. Indeed,
#     the ``apply()`` method is not necessarily defined, in the case of implicitly defined functionals.
#     The key method is ``grad()``, necessary to perform gradient flow optimization (and also used by
#     ``prox()``).
#
#     **Remark 2**
#
#     The ``apply()`` method raises a ``NotImplementedError`` unless the diffusion term is known to derive from
#     a variational formulation. Currently, only the case where :math:`\mathbf{D}_{in}` is a
#     :py:class:`~pyxu.operator.diffusion.DiffusionCoeffIsotropic` and all other diffusion coefficients/diffusivities
#     are `None` may detect an underlying variational formulation. Other cases exist but are not treated for now.
#
#     **Remark 3**
#
#     This class is not meant to be directly used, hence the underscore ``_DiffusionOp`` signalling it is private.
#     In principle, users should rely on the daughter classes :py:class:`~pyxu.operator.diffusion.DivergenceDiffusionOp`,
#     :py:class:`~pyxu.operator.diffusion.SnakeDiffusionOp`, :py:class:`~pyxu.operator.diffusion.TraceDiffusionOp`,
#     :py:class:`~pyxu.operator.diffusion.CurvaturePreservingDiffusionOp`.
#
#
#     Developer Notes
#     ---------------
#     * In method ``grad()``, to avoid using the @vectorize decorator, all ``_compute()`` functions should be changed, suitably stacking
#       all operators and results along the stacking dimensions. For now this has not been done, to be discussed if less naif
#       vectorisation is important. It would be cumbersome especially for the terms involving diffusion coefficients, whose
#       ``apply()`` method returns a Pyxu operator.
#
#     * For now, user is meant to initialize independently all the building blocks and provide them at initialization of a
#       diffusion operator. We could provide, of course, simpler interfaces for some of the most standard diffusion operators.
#       Still, even in current form, module should hopefully be relatively simple to use when provided with some examples.
#     """
#
#     def __init__(
#         self,
#         dim_shape: pyct.NDArrayShape,
#         nchannels: pyct.Integer = 1,
#         gradient: pyct.OpT = None,
#         hessian: pyct.OpT = None,
#         outer_diffusivity: pyct.OpT = None,
#         diffusion_coefficient: pyct.OpT = None,
#         balloon_force: pyct.OpT = None,
#         outer_trace_diffusivity: pyct.OpT = None,
#         trace_diffusion_coefficient: pyct.OpT = None,
#         curvature_preservation_field: pyct.NDArray = None,
#         prox_sigma: pyct.Real = 2,
#     ):
#         r"""
#
#         Parameters
#         ----------
#         dim_shape: pyct.Shape
#             Shape of the input array.
#         gradient: :py:class:`~pyxu.operator.linop.diff.Gradient`
#             Gradient operator. Defaults to `None`.
#         hessian:  :py:class:`~pyxu.operator.linop.diff.Hessian`
#             Hessian operator. Defaults to `None`.
#         outer_diffusivity: :py:class:`~pyxu.operator.diffusion._Diffusivity`
#             Outer diffusivity operator of the divergence term.
#         diffusion_coefficient: :py:class:`~pyxu.operator.diffusion._DiffusionCoefficient`
#             Diffusion coefficient operator of the divergence term.
#         balloon_force: :py:class:`~pyxu.operator.diffusion._BalloonForce`
#             Balloon force operator.
#         outer_trace_diffusivity: :py:class:`~pyxu.operator.diffusion._Diffusivity`
#             Outer diffusivity operator of the trace term.
#         trace_diffusion_coefficient: :py:class:`~pyxu.operator.diffusion._DiffusionCoefficient`
#             Diffusion coefficient operator of the trace term.
#         curvature_preservation_field: pyct.NDArray
#             Vector field along which curvature should be preserved. Defaults to `None`.
#         prox_sigma: pyct.Real
#             Size of the structures that should be smoothed out by prox computation (when implemented in PnP-fashion).
#
#         Notes
#         ----
#         The parameter ``prox_sigma`` is expressed in pixel units. Independently of the ``sampling`` of the differential
#         operators involved, ``prox_sigma`` equal to `1` is meant to yield a prox which smoothes structures of the order
#         of one pixel.
#         """
#         self.dim_shape = dim_shape
#         self.ndims = len(self.dim_shape)
#         self.nchannels = nchannels
#         super().__init__(shape=(1, int(np.prod(dim_shape)) * self.nchannels))
#         # super().__init__(shape=(1, int(np.prod(dim_shape))))
#         # sanitize inputs
#         (
#             gradient,
#             hessian,
#             diffusion_coefficient,
#             trace_diffusion_coefficient,
#             curvature_preservation_field,
#             sampling,
#         ) = self._sanitize_init_args(
#             dim_shape=dim_shape,
#             gradient=gradient,
#             hessian=hessian,
#             outer_diffusivity=outer_diffusivity,
#             diffusion_coefficient=diffusion_coefficient,
#             balloon_force=balloon_force,
#             outer_trace_diffusivity=outer_trace_diffusivity,
#             trace_diffusion_coefficient=trace_diffusion_coefficient,
#             curvature_preservation_field=curvature_preservation_field,
#             prox_sigma=prox_sigma,
#         )
#         self.outer_diffusivity = outer_diffusivity
#         self.diffusion_coefficient = diffusion_coefficient
#         self.balloon_force = balloon_force
#         self.outer_trace_diffusivity = outer_trace_diffusivity
#         self.trace_diffusion_coefficient = trace_diffusion_coefficient
#         self.curvature_preservation_field = curvature_preservation_field
#         if curvature_preservation_field is not None:
#             # compute jacobian of the field and apply it to field itself
#             self.jacobian = gradient(curvature_preservation_field)
#             ops = []
#             for i in range(self.ndims):
#                 vec = 0
#                 for j in range(self.ndims):
#                     vec += self.jacobian[i, self.dim * j : self.dim * (j + 1)] * curvature_preservation_field[j, :]
#                 ops.append(pybase.DiagonalOp(vec))
#             self._jacobian_onto_field = pyblock.hstack(ops)
#         # assess whether diffusion operator descends from a potential formulation or not
#         if self.diffusion_coefficient is not None:
#             self.from_potential = (
#                 self.diffusion_coefficient.from_potential
#                 and (self.outer_diffusivity is None)
#                 and (self.balloon_force is None)
#                 and (self.outer_trace_diffusivity is None)
#                 and (self.trace_diffusion_coefficient is None)
#                 and (self.curvature_preservation_field is None)
#             )
#         self.sampling = sampling
#         self.gradient = gradient
#         self.hessian = hessian
#         # estimate number of prox steps necessary to smooth structures of size prox_sigma (linear diffusion analogy)
#         self.prox_sigma = prox_sigma
#         t_final = self.prox_sigma**2 / 2
#         self.time_step = 1.0 / (2**self.ndims)
#         self.prox_steps = t_final / self.time_step
#         # set lipschitz and diff_lipschitz to np.inf
#         # lipschitz: think further, when apply exists we may have bounds on it. not crucial.
#         self._lipschitz = np.inf
#         self._diff_lipschitz = np.inf
#
#     def _sanitize_init_args(
#         self,
#         dim_shape: pyct.NDArrayShape,
#         gradient: pyct.OpT,
#         hessian: pyct.OpT,
#         outer_diffusivity: pyct.OpT,
#         diffusion_coefficient: pyct.OpT,
#         balloon_force: pyct.OpT,
#         outer_trace_diffusivity: pyct.OpT,
#         trace_diffusion_coefficient: pyct.OpT,
#         curvature_preservation_field: pyct.NDArray,
#         prox_sigma: pyct.Real,
#     ):
#         if hessian is not None:
#             nb_upper_entries = round(self.ndims * (self.ndims + 1) / 2)
#             expected_codim = nb_upper_entries * self.dim
#             assert hessian.codim == expected_codim, '`hessian` expected to be initialized with `directions`="all"'
#
#         if outer_diffusivity is not None and diffusion_coefficient is None:
#             raise ValueError("Cannot provide `outer_diffusivity` without providing `diffusion_coefficient`.")
#
#         if outer_trace_diffusivity is not None and trace_diffusion_coefficient is None:
#             raise ValueError(
#                 "Cannot provide `outer_trace_diffusivity` without providing `trace_diffusion_coefficient`."
#             )
#
#         if (
#             (diffusion_coefficient is None)
#             and (balloon_force is None)
#             and (trace_diffusion_coefficient is None)
#             and (curvature_preservation_field is None)
#         ):
#             msg = "\n".join(
#                 [
#                     "Cannot instantiate the diffusion operator. Pass at least one of the following:",
#                     "`diffusion_coefficient`, `balloon_force`, `trace_diffusion_coefficient`, `curvature_preservation_field`.",
#                 ]
#             )
#             raise ValueError(msg)
#
#         if diffusion_coefficient is not None and gradient is None:
#             msg = "\n".join(
#                 [
#                     "No`gradient` was passed, needed for divergence term involving `diffusion_coefficient`.",
#                     "Initializing a forward finite difference operator with unitary sampling as default.",
#                 ]
#             )
#             warnings.warn(msg)
#             gradient = pydiff.Gradient(
#                 dim_shape=dim_shape,
#                 diff_method="fd",
#                 sampling=1.0,
#                 mode="symmetric",
#                 scheme="forward",
#             )
#
#         if curvature_preservation_field is not None and gradient is None:
#             msg = "\n".join(
#                 [
#                     "No `gradient` was passed, needed for term involving `curvature_preservation_field`.",
#                     "Initializing a central finite difference operator with unitary sampling as default.",
#                 ]
#             )
#             warnings.warn(msg)
#             gradient = pydiff.Gradient(
#                 dim_shape=dim_shape, diff_method="fd", sampling=1.0, mode="edge", scheme="central"
#             )
#
#         if trace_diffusion_coefficient is not None and hessian is None:
#             msg = "\n".join(
#                 [
#                     "No `hessian` was passed, needed for trace term involving `trace_diffusion_coefficient`.",
#                     "Initializing a central finite difference operator with unitary sampling as default.",
#                 ]
#             )
#             warnings.warn(msg)
#             hessian = pydiff.Hessian(
#                 dim_shape=dim_shape, diff_method="fd", mode="symmetric", sampling=1.0, scheme="central", accuracy=2
#             )
#
#         if diffusion_coefficient is not None and diffusion_coefficient.trace_term:
#             if not diffusion_coefficient.frozen:
#                 warnings.warn("`diffusion_coefficient.trace_term` set to True. Modifying to False.")
#                 diffusion_coefficient.trace_term = True
#             else:
#                 msg = "\n".join(
#                     [
#                         "`diffusion_coefficient.trace_term` set to True and `diffusion_coefficient.frozen` set to True.",
#                         "Issues are expected. Initialize correctly `diffusion_coefficient.trace_term` to False before freezing.",
#                     ]
#                 )
#                 raise ValueError(msg)
#
#         if trace_diffusion_coefficient is not None and not trace_diffusion_coefficient.trace_term:
#             if not trace_diffusion_coefficient.frozen:
#                 warnings.warn("`trace_diffusion_coefficient.trace_term` set to False. Modifying to True.")
#                 trace_diffusion_coefficient.trace_term = True
#             else:
#                 msg = "\n".join(
#                     [
#                         "`trace_diffusion_coefficient.trace_term` set to False while `trace_diffusion_coefficient.frozen` set to True.",
#                         "Issues are expected. Initialize correctly `trace_diffusion_coefficient.trace_term` to True before freezing.",
#                     ]
#                 )
#                 raise ValueError(msg)
#
#         if curvature_preservation_field is not None:
#             if curvature_preservation_field.shape != (self.ndims, self.dim):
#                 msg = "\n".join(
#                     [
#                         "Unexpected shape {} of `curvature_preservation_field`,"
#                         "expected ({}, {}).".format(curvature_preservation_field.shape, self.ndims, self.dim),
#                     ]
#                 )
#                 raise ValueError(msg)
#
#         # check dim_shapes consistency
#         _to_be_checked = {
#             "outer_diffusivity": outer_diffusivity,
#             "diffusion_coefficient": diffusion_coefficient,
#             "balloon_force": balloon_force,
#             "outer_trace_diffusivity": outer_trace_diffusivity,
#             "trace_diffusion_coefficient": trace_diffusion_coefficient,
#             "gradient": gradient,
#             "hessian": hessian,
#         }
#         for i in _to_be_checked:
#             if _to_be_checked[i]:
#                 msg = "`{}.dim_shape`=({}) inconsistent with `dim_shape`={}.".format(
#                     i, _to_be_checked[i].dim_shape, dim_shape
#                 )
#                 assert _to_be_checked[i].dim_shape == dim_shape, msg
#
#         # check sampling consistency
#         _to_be_checked = {}
#         # if gradient:
#         #    _to_be_checked["`gradient`"] = gradient.sampling
#         # if hessian:
#         #     _to_be_checked["`hessian`"] = hessian.sampling
#         # if balloon_force:
#         #     if balloon_force.gradient:
#         #         _to_be_checked["`balloon_force.gradient`"] = balloon_force.gradient.sampling
#         # if outer_diffusivity:
#         #     if outer_diffusivity.gradient:
#         #         _to_be_checked["`outer_diffusivity.gradient`"] = outer_diffusivity.gradient.sampling
#         # if outer_trace_diffusivity:
#         #     if outer_trace_diffusivity.gradient:
#         #         _to_be_checked["`outer_trace_diffusivity.gradient`"] = outer_trace_diffusivity.gradient.sampling
#         # if diffusion_coefficient:
#         #     if diffusion_coefficient.isotropic:
#         #         if diffusion_coefficient.diffusivity.gradient:
#         #             _to_be_checked[
#         #                 "`diffusion_coefficient.diffusivity.gradient`"
#         #             ] = diffusion_coefficient.diffusivity.gradient.sampling
#         #     else:
#         #         if diffusion_coefficient.structure_tensor:
#         #             _to_be_checked[
#         #                 "`diffusion_coefficient.structure_tensor.gradient`"
#         #             ] = diffusion_coefficient.structure_tensor.grad.sampling
#         # if trace_diffusion_coefficient:
#         #     if trace_diffusion_coefficient.isotropic:
#         #         if trace_diffusion_coefficient.diffusivity.gradient:
#         #             _to_be_checked[
#         #                 "`trace_diffusion_coefficient.diffusivity.gradient`"
#         #             ] = trace_diffusion_coefficient.diffusivity.gradient.sampling
#         #     else:
#         #         if trace_diffusion_coefficient.structure_tensor:
#         #             _to_be_checked[
#         #                 "`trace_diffusion_coefficient.structure_tensor.gradient`"
#         #             ] = trace_diffusion_coefficient.structure_tensor.grad.sampling
#         # if _to_be_checked:
#         #     s_base = list(_to_be_checked.values())[0]
#         #     op_base = list(_to_be_checked.keys())[0]
#         #     for s in _to_be_checked:
#         #         assert (
#         #             _to_be_checked[s] == s_base
#         #         ), "Inconsistent `sampling` for differential operators {} and {}.".format(op_base, s)
#         #     sampling = s_base
#         # else:
#         #     sampling = None
#         sampling = 1.0
#
#         assert prox_sigma > 0.0, "`prox_sigma` must be strictly positive."
#
#         # if trace_diffusion_coefficient is isotropic,
#         # convert hessian to second derivative operator
#         if trace_diffusion_coefficient is not None:
#             if trace_diffusion_coefficient.isotropic:
#                 ops = []
#                 idx = 0
#                 for dim in range(self.ndims):
#                     # select second order derivative operators
#                     ops.append(hessian._block[(idx, 0)])
#                     idx += self.ndims - dim
#                 hessian = pyblock.vstack(ops)
#                 hessian = pydiff._make_unravelable(hessian, dim_shape=dim_shape)
#
#         # returning only objects that might have been modified.
#         return (
#             gradient,
#             hessian,
#             diffusion_coefficient,
#             trace_diffusion_coefficient,
#             curvature_preservation_field,
#             sampling,
#         )
#
#     def asloss(self, data: pyct.NDArray = None) -> NotImplemented:
#         """
#         Notes
#         -------
#         DivergenceDiffusionOp class is not meant to be used to define a loss functional.
#         """
#         return NotImplemented
#
#     #@pycrt.enforce_precision(i="arr")
#     @pycu.vectorize("arr")
#     def apply(self, arr: pyct.NDArray) -> NotImplemented:
#         r"""
#         Notes
#         -------
#         Divergence-based diffusion operators may arise from a variational formulation. This is true, e.g.,
#         for the isotropic Perona-Malik, TV, Tikhonov. For these cases, it is possible
#         to define the associated energy functional. When no variational formulation is detected, the method raises an error.
#         """
#         if self.from_potential:
#             return self.diffusion_coefficient.diffusivity.energy_functional(arr, self.gradient)
#         else:
#             msg = "\n".join(
#                 [
#                     "DivergenceDiffusionOp not found to be arising from an energy potential formulation.",
#                     "If it is, define how to evaluate the associated energy functional.",
#                 ]
#             )
#             raise NotImplementedError(msg)
#
#     #@pycrt.enforce_precision(i="arr")
#     def _compute_divergence_term(self, arr: pyct.NDArray) -> pyct.NDArray:
#         if self.diffusion_coefficient is not None or self.outer_diffusivity is not None:
#             y_div = self.gradient(arr)
#             if self.diffusion_coefficient is not None:
#                 # compute flux
#                 diffusion_coefficient = self.diffusion_coefficient(arr)
#                 y_div = diffusion_coefficient(y_div)
#             # apply divergence
#             y_div = self.gradient.T(y_div)
#             if self.outer_diffusivity is not None:
#                 outer_diffusivity = self.outer_diffusivity(arr)
#                 # rescale divergence
#                 y_div *= outer_diffusivity
#         else:
#             xp = pycu.get_array_module(arr)
#             y_div = xp.zeros_like(arr)
#         return y_div
#
#     #@pycrt.enforce_precision(i="arr")
#     def _compute_balloon_term(self, arr: pyct.NDArray) -> pyct.NDArray:
#         if self.balloon_force is not None:
#             balloon_force = self.balloon_force(arr)
#         else:
#             xp = pycu.get_array_module(arr)
#             balloon_force = xp.zeros_like(arr)
#         return -balloon_force
#
#     #@pycrt.enforce_precision(i="arr")
#     def _compute_trace_term(self, arr: pyct.NDArray) -> pyct.NDArray:
#         if self.trace_diffusion_coefficient is not None:
#             # hessian = self.hessian.unravel(self.hessian(arr)).squeeze().reshape(1, -1)
#             hessian = self.hessian.unravel(self.hessian(arr)).reshape(self.nchannels, -1)
#             trace_tensor = self.trace_diffusion_coefficient(arr)
#             y_trace = trace_tensor(hessian)
#             if self.outer_trace_diffusivity is not None:
#                 outer_trace_diffusivity = self.outer_trace_diffusivity(arr)
#                 # rescale trace
#                 y_trace *= outer_trace_diffusivity(arr)
#         else:
#             xp = pycu.get_array_module(arr)
#             y_trace = xp.zeros_like(arr)
#         return -y_trace
#
#     #@pycrt.enforce_precision(i="arr")
#     def _compute_curvature_preserving_term(self, arr: pyct.NDArray) -> pyct.NDArray:
#         if self.curvature_preservation_field is not None:
#             grad_arr = self.gradient(arr)
#             y_curv = self._jacobian_onto_field(grad_arr)
#         else:
#             xp = pycu.get_array_module(arr)
#             y_curv = xp.zeros_like(arr)
#         return -y_curv
#
#     #@pycrt.enforce_precision(i="arr")
#     @pycu.vectorize("arr")
#     def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
#         arr = arr.reshape(self.nchannels, -1)
#         # arr = arr.reshape(1, -1)
#         # compute divergence term
#         y = self._compute_divergence_term(arr)
#         # compute balloon force term
#         y += self._compute_balloon_term(arr)
#         # compute trace tensor term
#         y += self._compute_trace_term(arr)
#         # compute curvature preserving term
#         y += self._compute_curvature_preserving_term(arr)
#         return y.reshape(1, -1)
#
#     #@pycrt.enforce_precision(i="arr")
#     @pycu.vectorize("arr")
#     def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
#         r"""
#
#         Notes
#         -----
#         Let :math:`\phi(\cdot)` be the functional (possibly defined only implicitly via its gradient) underlying
#         the diffusion operator. Its prox operator is defined as
#
#         .. math::
#             \mathrm{prox}_{\tau \phi}(\mathbf{x}) = \underset{\mathbf{y}}{\mathrm{argmin}} \frac{1}{2\tau}\Vert \mathbf{x} - \mathbf{y}\Vert _2^2+\phi(\mathbf{y}).
#
#         The prox can be interpreted as a denoising operator [see `Romano <https://arxiv.org/pdf/1611.02862.pdf>`_],
#         where the data-fidelity term :math:`\frac{1}{2\tau}\Vert \mathbf{x} - \mathbf{y}\Vert _2^2`
#         ensures that the result stays close to :math:`\mathbf{x}` and the regularisation term :math:`\phi(\mathbf{y})`
#         smooths the image.
#
#         In the spirit of Plug&Play (PnP) approaches [see `Romano <https://arxiv.org/pdf/1611.02862.pdf>`_], we
#         replace the solution of the prox problem with a denoiser that consists in performing a fixed number of
#         gradient-descent steps to solve the prox problem, which only requires evaluating its gradient
#
#         .. math::
#             \frac{1}{\tau}(\mathbf{y} - \mathbf{x})-\nabla\phi(\mathbf{y}).
#
#         This approach allows us to:
#
#         * bypass the problem of the explicit definition of the functional :math:`\phi(\cdot)` (only its gradient
#           :math:`\nabla\phi(\cdot)` is needed);
#         * have a prox operator that can be evaluated at a fixed cost (the number of chosen ``grad()`` calls,
#           i.e., ``prox_steps``), independent of the number of iterations required to converge to the true prox solution.
#
#         This denoising approach relies on the scale-space interpretation of diffusion operators
#         [see `Weickert <https://www.mia.uni-saarland.de/weickert/Papers/book.pdf>`_], according to which
#         denoising at a characteristic noise scale :math:`\sigma` can be achieved by stopping the diffusion
#         at a given time :math:`T`. In the linear isotropic diffusion case where the gradient of the diffusion operator
#         is the Laplacian, smoothing structures of scale :math:`\sigma` is achieved stopping the diffusion
#         process at :math:`T=\frac{\sigma^2}{2}`. Following the linear diffusion analogy,
#         the stopping time for the prox computation is given by :math:`T=\frac{\mathrm{prox\_sigma}^2}{2}`,
#         where ``prox_sigma`` is provided at initialization. Better estimates of stopping time
#         could/should be studied.
#         """
#         stop_crit = pystop.MaxIter(self.prox_steps)
#         pgd = pysol.PGD(f=self, g=None, show_progress=False, verbosity=100)
#         pgd.fit(**dict(mode=pysolver.Mode.BLOCK, x0=arr, stop_crit=stop_crit, acceleration=False))
#         return pgd.solution()
