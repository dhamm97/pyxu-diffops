import numpy as np
import pyxu.abc as pyca
import pyxu.info.ptype as pyct
import pyxu.util as pycu

__all__ = ["_ExtraDiffusionTerm", "Dilation", "Erosion", "MfiExtraTerm", "CurvaturePreservingTerm"]


class _ExtraDiffusionTerm(pyca.Map):
    r"""
    Abstract balloon force operator. Daughter classes implement specific balloon force terms.
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
        self.ndims = len(dim_shape) - 1
        self.nchannels = dim_shape[0]
        super().__init__(dim_shape=dim_shape, codim_shape=dim_shape)
        self.gradient = gradient
        if gradient is not None:
            msg = "`gradient.dim_shape`={} inconsistent with `dim_shape`={}.".format(gradient.dim_shape, dim_shape)
            assert gradient.dim_shape == dim_shape, msg

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        raise NotImplementedError


class Dilation(_ExtraDiffusionTerm):
    r"""
    Dilation operator.

    Notes
    -----
    Given a :math:`D`-dimensional signal:

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0} \times \cdots \times N_{D-1}},

    its ``apply()`` method computes

    .. math::

        \vert \nabla \mathbf{f} \vert \in \mathbb{R}^{N_{0} \times \cdots \times N_{D-1}},

    where :math:`\vert\cdot\vert` is the :math:`L^2`-norm of the gradient (on each pixel).

    Can be used to obtain the PDE version of the morphological dilation operator, which reads

    .. math::

        \frac{\partial \mathbf{f}}{\partial t}=\vert \nabla \mathbf{f}\vert.
    """

    def __init__(self, dim_shape: pyct.NDArrayShape, gradient: pyct.OpT):
        """

        Parameters
        ----------
        dim_shape: tuple
            Shape of the input array.
        gradient: :py:class:`~pyxu.operator.linop.diff.Gradient`
            Gradient operator.
        """
        super().__init__(dim_shape=dim_shape, gradient=gradient)

    # @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
        z = self.gradient(arr)  # (batch,ndims,nchannels,nx,ny)
        z **= 2
        z = xp.sum(z, axis=(-4, -3), keepdims=False)  # (batch,nx,ny)
        z = xp.sqrt(z)
        out = xp.stack([z] * self.nchannels, axis=-3)  # (batch,nchannels,nx,ny)
        return out


class Erosion(_ExtraDiffusionTerm):
    r"""
    Erosion operator.

    Notes
    -----
    Given a :math:`D`-dimensional signal:

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0} \times \cdots \times N_{D-1}},

    its ``apply()`` method computes

    .. math::

        - \vert \nabla \mathbf{f} \vert \in \mathbb{R}^{N_{0} \times \cdots \times N_{D-1}},

    where :math:`\vert\cdot\vert` is the :math:`L^2`-norm of the gradient (on each pixel).

    Can be used to obtain the PDE version of the morphological dilation operator, which reads

    .. math::

        \frac{\partial \mathbf{f}}{\partial t}=-\vert \nabla \mathbf{f}\vert.
    """

    def __init__(self, dim_shape: pyct.NDArrayShape, gradient: pyct.OpT):
        """

        Parameters
        ----------
        dim_shape: tuple
            Shape of the input array.
        gradient: :py:class:`~pyxu.operator.linop.diff.Gradient`
            Gradient operator.
        """
        super().__init__(dim_shape=dim_shape, gradient=gradient)

    # @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
        z = self.gradient(arr)  # (batch,ndims,nchannels,nx,ny)
        z **= 2
        z = xp.sum(z, axis=(-4, -3), keepdims=False)  # (batch,nx,ny)
        z = xp.sqrt(z)
        out = xp.stack([z] * self.nchannels, axis=-3)  # (batch,nchannels,nx,ny)
        return -out


class MfiExtraTerm(_ExtraDiffusionTerm):
    def __init__(
        self,
        dim_shape: pyct.NDArrayShape,
        gradient: pyct.OpT,
        beta: pyct.Real = 1,
        clipping_value: pyct.Real = 1e-5,
        tame: bool = True,
    ):
        """

        Parameters
        ----------
        dim_shape: tuple
            Shape of the input array.
        gradient: :py:class:`~pyxu.operator.linop.diff.Gradient`
            Gradient operator.
        clipping_value: pyct.Real
            Contrast parameter. Defaults to `1e-5`.
        tame: bool
            Whether to consider the tame version bounded in :math:`(0, 1]` or not. Defaults to `True`.
        """
        super().__init__(dim_shape=dim_shape, gradient=gradient)
        self.beta = beta
        self.clipping_value = clipping_value
        self.tame = tame
        if tame:
            self.lipschitz = 1
        else:
            self.lipschitz = 1 / clipping_value

    # @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
        y = self.gradient(arr)  # (batch,ndims,nchannels,nx,ny)
        y **= 2
        y = xp.sum(y, axis=-4, keepdims=False)  # (batch,nchannels,nx,ny)
        z = xp.clip(arr, self.clipping_value, None)
        if self.tame:
            z /= self.beta
            z += 1
        z **= -2
        z /= 2
        if self.tame:
            z /= self.beta
        return y * z  # (batch,nchannels,nx,ny)


class CurvaturePreservingTerm(_ExtraDiffusionTerm):
    def __init__(self, dim_shape: pyct.NDArrayShape, gradient: pyct.OpT, curvature_preservation_field: pyct.NDArray):
        """

        Parameters
        ----------
        dim_shape: tuple
            Shape of the input array.
        gradient: :py:class:`~pyxu.operator.linop.diff.Gradient`
            Gradient operator. Recommended settings: mode="edge", scheme="central"
        curvature_preservation_field: pyct.NDArray
            Vector field along which curvature should be preserved.

        """
        super().__init__(dim_shape=dim_shape, gradient=gradient)
        # self.ndims = len(dim_shape)
        if curvature_preservation_field.shape != (self.ndims, *dim_shape[1:]):
            msg = "\n".join(
                [
                    "Unexpected shape {} of `curvature_preservation_field`,"
                    "expected ({}, {}).".format(curvature_preservation_field.shape, self.ndims, dim_shape[1:]),
                ]
            )
            raise ValueError(msg)
        xp = pycu.get_array_module(curvature_preservation_field)
        self.curvature_preservation_field = curvature_preservation_field  # (2dim,nx,ny)
        # compute jacobian of the field and apply it to field itself
        self.jacobian = gradient(xp.expand_dims(curvature_preservation_field, 1)).squeeze()  # (2dim,2dimgrad,nx,ny)
        # assemble operator involved in curvature preserving term
        self._jacobian_onto_field = xp.zeros((self.ndims, *dim_shape[1:]))  # (2dim,nx,ny)
        for i in range(self.ndims):
            vec = 0
            for j in range(self.ndims):
                vec += self.jacobian[i, j, :, :] * curvature_preservation_field[j, :, :]

            self._jacobian_onto_field[i, :, :] = vec  # (2dim,nx,ny)

        # compute lipschitz constant of curvature preserving term
        self.max_norm = np.max(np.linalg.norm(curvature_preservation_field, axis=0))
        # abs(<gradient(u), J_w(w)>) \leq norm(gradient(u)) * norm(J_w(w))
        # \leq L_grad*norm(u)*2*L_grad*(norm(w)**2) = 2*L_grad**2 * norm(w)**2 * norm(u)
        self.lipschitz = 2 * (gradient.lipschitz**2) * self.max_norm**2

    # @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)  # (batch,nchannels,nx,ny)
        grad_arr = self.gradient(arr)  # (batch,2dimgrad,nchannels,nx,ny)
        grad_arr = xp.moveaxis(grad_arr, -3, -4)  # (batch,nchannels,2dimgrad,nx,ny)
        y_curv = xp.einsum(
            "...ijk,...ijk->...jk",  # inner product jacfield-grad for every pixel
            xp.reshape(
                self._jacobian_onto_field, (*([1] * len(grad_arr.shape[:-3])), *self._jacobian_onto_field.shape)
            ),  # (1,1,2dimgrad,nx,ny) adding dummy dimension
            grad_arr,
        )
        return -y_curv  # (batch,nchannels,nx,ny)
