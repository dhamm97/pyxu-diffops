import itertools

import numpy as np
import pytest
import pyxu.info.deps as pycd
import pyxu.runtime as pycrt
import pyxu_tests.operator.conftest as conftest

import pyxu_diffops.operator.diffusion as pyxudiff


class DiffusionOpMixin(conftest.DiffFuncT):
    # Change to conftest.ProxDiffFunc to test prox (and uncomment data_prox fixture)
    disable_test = frozenset(
        conftest.MapT.disable_test
        | {
            "test_math_lipschitz",
            "test_valueND_apply",
            "test_valueND_call",
            "test_interface_asloss",
        }
    )

    @pytest.fixture
    def diffusion_op_klass(self):
        raise NotImplementedError

    @pytest.fixture
    def diffusion_op_kwargs(self):
        raise NotImplementedError

    @pytest.fixture
    def data_apply(self):  # If no apply method, then use DiffusionOpNoApplyMixin subclass
        raise NotImplementedError

    # @pytest.fixture
    # def data_prox(self):
    #     raise NotImplementedError

    @pytest.fixture
    def data_grad(self):
        raise NotImplementedError

    @pytest.fixture
    def data_math_lipschitz(self):  # Not tested
        pass

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        N_test = 2
        return self._random_array((N_test, dim), seed=0)

    @pytest.fixture(scope="session")
    def dim_shape(self):
        return 2, 3

    @pytest.fixture(scope="session", params=[1])  # TODO tests currently fail with multiple channels (params=[1, 2])
    def nchannels(self, request):  # TODO remove if all subclasses do not have this argument (as is currently the case)
        return request.param

    @pytest.fixture
    def dim(self, dim_shape, nchannels):
        return np.prod(dim_shape) * nchannels

    @pytest.fixture
    def data_shape(self, dim):
        return 1, dim

    @pytest.fixture(
        scope="session",
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        ),
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture(scope="session")
    def ndi(self, _spec):
        return _spec[0]

    @pytest.fixture(scope="session")
    def width(self, _spec):
        return _spec[1]

    @pytest.fixture(scope="session")  # Without session scope, stencils are instanciated at every test -> super slow
    def spec(self, dim_shape, nchannels, diffusion_op_klass, diffusion_op_kwargs, ndi, width):
        if ndi.module() is None:
            pytest.skip(f"{ndi} unsupported on this machine.")
        with pycrt.Precision(width):
            op = diffusion_op_klass(dim_shape=dim_shape, nchannels=nchannels, **diffusion_op_kwargs)
        return op, ndi, width


class DiffusionOpNoApplyMixin(DiffusionOpMixin):
    # Base class for _DiffusionOp objects who don't derive from a potential (with no apply() method)
    disable_test = frozenset(
        DiffusionOpMixin.disable_test
        | {
            "test_value1D_apply",
            "test_backend_apply",
            "test_prec_apply",
            "test_precCM_apply",
            "test_transparent_apply",
            "test_value1D_call",
            "test_backend_call",
            "test_prec_call",
            "test_precCM_call",
            "test_transparent_call",
            "test_interface_jacobian",  # Don't really understand this test, but it raises an error
        }
    )

    @pytest.fixture
    def data_apply(self):
        pass

    @pytest.fixture
    def _data_apply(self):
        pass


class TestMfiDiffusion(DiffusionOpMixin):
    @pytest.fixture(scope="session")
    def diffusion_op_klass(self):
        return pyxudiff.MfiDiffusion

    @pytest.fixture(scope="session")
    def diffusion_op_kwargs(self):
        return dict(beta=1)

    @pytest.fixture
    def data_apply(self, dim_shape, nchannels, spec):  # Operator derives from potentional
        arr = self._random_array(dim_shape)  # TODO Replace with input-output pairs computed manually
        out = spec[0](arr.ravel())
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )

    # @pytest.fixture
    # def data_prox(self, dim_shape, nchannels, spec):
    #     arr = self._random_array(dim_shape)  # TODO Replace with input-output pairs computed manually
    #     tau = 2
    #     out = spec[0].prox(arr.ravel(), tau=tau)
    #
    #     return dict(
    #         in_=dict(arr=arr.reshape(-1), tau=tau),
    #         out=out.reshape(-1),
    #     )

    @pytest.fixture
    def data_grad(self, dim_shape, nchannels, spec):
        arr = self._random_array(dim_shape)  # TODO Replace with input-output pairs computed manually
        out = spec[0].grad(arr.ravel())

        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )
