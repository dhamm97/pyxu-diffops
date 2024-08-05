import pyxu.abc as pxa
import pyxu.info.ptype as pxt

__all__ = ["NullFunc"]


def NullFunc(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    """
    Null functional (modified from base Pyxu class).
    This functional maps any input vector on the null scalar.

    The plugin modification adds a print at init time.
    """
    from pyxu.operator.linop.base import NullOp

    op = NullOp(
        dim_shape=dim_shape,
        codim_shape=1,
    ).asop(pxa.LinFunc)
    op._name = "NullFunc"
    op._name = "ModifiedNullFunc"
    print(
        "The modified NullFunc exemplifies how to overload a base class. ",
        "To overload a Pyxu base class, an underscore needs to be added in front of the class name, "
        "in the pyproject.toml file section [entry_points]).",
    )
    return op
