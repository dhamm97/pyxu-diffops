API Reference
=============


.. The goal of this page is to provide an alphabetical listing of all pyxu-diffops objects exposed to users.
.. This is achieved via the `autosummary` extension.
.. While `autosummary` understands `automodule`-documented packages, explicitly listing a module's contents is required.

Welcome to the official API reference for pyxu-diffops.
This API documentation is intended to serve as a comprehensive guide to the library's various modules, classes, functions, and interfaces.
It provides detailed descriptions of each component's role, relations, assumptions, and behavior.

To use this plugin, users can instantiate the class corresponding to the desired diffusion operator. Such classes rely on private modules
implementing the building blocks composing each operator; only advanced users wanting to implement new operators should look into them.
The code for such private modules is available on the repository but no compiled documentation is provided.
All diffusion operators are daughter classes of the private base class :py:class:`~pyxu_diffops.operator._Diffusion`.

**Remark**

Diffusion operators are implemented as differentiable functionals :py:class:`~pyxu.abc.operator.DiffFunc`.
However, they are atypical differentiable functionals. Indeed,
the ``apply()`` method is not necessarily defined, in the case of implicitly defined functionals.
The key method is ``grad()``, necessary to perform gradient flow optimization. The ``apply()`` method raises a ``NotImplementedError`` unless the diffusion term is known to derive from
a variational formulation.

The Pyxu-diffops plug-in implements the following diffusion operators.

Diffusion operators
-----------

pyxu_diffops.operator
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

   ~pyxu_diffops.operator.MfiDiffusion
   ~pyxu_diffops.operator.PeronaMalikDiffusion
   ~pyxu_diffops.operator.TikhonovDiffusion
   ~pyxu_diffops.operator.TotalVariationDiffusion
   ~pyxu_diffops.operator.CurvaturePreservingDiffusionOp
   ~pyxu_diffops.operator.AnisEdgeEnhancingDiffusionOp
   ~pyxu_diffops.operator.AnisCoherenceEnhancingDiffusionOp
   ~pyxu_diffops.operator.AnisDiffusionOp


.. toctree::
   :maxdepth: 2
   :hidden:

   operator


