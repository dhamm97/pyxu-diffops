:html_theme.sidebar_secondary.remove:
:sd_hide_title: true

.. |br| raw:: html

   </br>

.. raw:: html

    <!-- CSS overrides on the homepage only -->
    <style>
    .bd-main .bd-content .bd-article-container {
    max-width: 70rem; /* Make homepage a little wider instead of 60em */
    }
    /* Extra top/bottom padding to the sections */
    article.bd-article section {
    padding: 2rem 0 5rem;
    }
    /* Override all h1 headers except for the hidden ones */
    h1:not(.sd-d-none) {
    font-size: 48px;
    text-align: left;
    margin-bottom: 4rem;
    }
    /* Override all h3 headers that are not in hero */
    h3:not(#hero h3) {
    font-weight: bold;
    text-align: left;
    }

    p {
    text-align: justify;
    }

    a:visited {
    color: var(--pst-color-primary);
    }

    .homepage-button.primary-button:visited {
    color: var(--pst-color-background);
    }


    .sponsors-list-item {
    display: inline-flex;
    justify-content: center;
    opacity: 0.5;
    filter: brightness(0.5) grayscale(1);
    }

    @keyframes platformsSlideshow {
    100% {
        transform: translateX(-2000px);
    }
    }
    </style>

.. raw:: html

    <div id="hero">

Pyxu_diffops
============

.. raw:: html

    <h2 style="font-size: 60px; font-weight: bold; display: inline"><span>Pyxu-diffops</span></h2>
    <p>
    Welcome to <strong> Pyxu-diffops</strong>, a Pyxu plugin implementing many diffusion operators which are popular in the PDE-based image processing community.
    Here, we provide a basic introduction to the PDE-based image processing techniques implemented by this plugin.
    </p>
    <div class="homepage-button-container">
    <div class="homepage-button-container-row">
        <a href="./examples/index.html" class="homepage-button primary-button">See Examples</a>
        <a href="./api/index.html" class="homepage-button secondary-button">See API Reference </a>
    </div>
    </div>
    </div>


Diffusion operators
===================

This plugin provides an interface to deal with PDE-based image processing [Weickert]_, [Tschumperle]_. For simplicity, throughout the
documentation we consider a :math:`2`-dimensional image :math:`\mathbf{f} \in \mathbb{R}^{N_{0} \times N_1}`,
but higher dimensional signals could be considered. We denote by :math:`f_i` the :math:`i`-th pixel
of :math:`\mathbf{f},\;i=0,\ldots,N_{tot}-1,\;N_{tot}=N_0N_1`. Furthermore, let
:math:`\boldsymbol{\nabla} \mathbf{f} \in \mathbb{R}^{2 \times N_{0} \times N_1}` be the image gradient; we denote by :math:`(\boldsymbol{\nabla} \mathbf{f})_i \in \mathbb{R}^{2}`
the local gradient in correspondence of the :math:`i`-th pixel.

Let us first consider the simple case of PDE-based image smoothing [Weickert]_. Gaussian smoothing, or linear diffusion filtering, can be achieved computing the time evolution
of the Partial Differential Equation (PDE)

.. math::
    \frac{\partial\mathbf{f}}{\partial t} = \mathrm{div}(\boldsymbol{\nabla}\mathbf{f})
                                          = \boldsymbol{\Delta} \mathbf{f},

where :math:`t` represents an artificial time and the divergence operator acts on the local gradient. The above **diffusion equation** corresponds to the
well-known heat equation. It achieves linear, isotropic, homogeneous smoothing. In practice, the time evolution can be computed
discretizing the PDE in time; if we use the simple explicit Euler method with time-step  :math:`\Delta t=\tau` , we obtain the scheme

.. math::
    \mathbf{f}_{n+1} = \mathbf{f}_n + \tau \mathrm{div}(\boldsymbol{\nabla}\mathbf{f}_n).
   :label: update_rule

Letting the PDE evolve in time, we obtain increasingly smoother versions of the original image. Eventually, we would reach
the steady state :math:`\frac{\partial\mathbf{f}}{\partial t}=0`, corresponding to a flat image. By early stopping, we can achieve
arbitrarily smooth versions of :math:`\mathbf{f}`.

In Gaussian smoothing, the divergence is applied to the flux :math:`\boldsymbol{\nabla} \mathbf{f}`, which coincides with the image gradient. However, the
flux can be locally modified to promote desired features by designing suitable **diffusion tensors** :math:`\mathbf{D}(\mathbf{f})`, leading to the diffusion equation

.. math::
    \frac{\partial\mathbf{f}}{\partial t} = \mathrm{div}(\mathbf{D}(\mathbf{f})\boldsymbol{\nabla}\mathbf{f}).
   :label: grad_flow_divdiff

The diffusion tensor acts on the local gradient, biasing the diffusion process to enhance features of interest. Anisotropic, inhomogeneous
smoothing can thus be achieved. Many diffusion tensors have been proposed. Furthermore, generalizations
going beyond purely divergence-based diffusion terms have been investigated [Tschumperle]_.

The Pyxu-diffops module allows to consider diffusion processes that, in their most general form, can be written as

.. math::
    \frac{\partial\mathbf{f}}{\partial t} = \mathbf{D}_{out}\mathrm{div}(\mathbf{D}_{in}\boldsymbol{\nabla}\mathbf{f})
    + \mathbf{b} + \mathbf{T}_{out}\mathrm{trace}\big(\mathbf{T}_{in}\mathbf{H}(\mathbf{f})\big) + (\boldsymbol{\nabla}\mathbf{f})^T \mathbf{J}_{\mathbf{w}}\mathbf{w},
   :label: grad_flow_generic

where
    * :math:`\mathbf{D}_{out} = \mathbf{D}_{out}(\mathbf{f})` is the outer diffusivity for the divergence term,
    * :math:`\mathbf{D}_{in} = \mathbf{D}_{in}(\mathbf{f})` is the diffusion coefficient for the divergence term,
    * :math:`\mathbf{b} = \mathbf{b}(\mathbf{f})` is the balloon force,
    * :math:`\mathbf{T}_{out} = \mathbf{T}_{out}(\mathbf{f})` is the outer diffusivity for the trace term,
    * :math:`\mathbf{T}_{in} = \mathbf{T}_{in}(\mathbf{f})` is the diffusion coefficient for the trace term,
    * :math:`\mathbf{H}(\mathbf{f})` is the Hessian evaluated at :math:`\mathbf{f}`,
    * :math:`\mathbf{w}` is a vector field assigning a :math:`2`-dimensional vector to each pixel,
    * :math:`\mathbf{J}_\mathbf{w}` is the Jacobian of the vector field :math:`\mathbf{w}`.

Pyxu-diffops offers a collection of diffusion operators, implemented as Pyxu differentiable functionals :py:class:`~pyxu.abc.operator.DiffFunc`. These classes
exhibit an ``apply()`` and a ``grad()`` method.

The right-hand side of the above PDE represents the output of the ``grad()`` method applied to the image :math:`\mathbf{f}`.

To understand why the diffusion operators are implemented as functionals, i.e., why they exhibit an ``apply()`` method, we need to need to consider PDE-based restoration rather than simple smoothing.

PDE-based restoration
---------------------

In some cases, the right-hand side of the PDE :eq:`grad_flow_generic` admits a potential, i.e., there exists a functional :math:`\phi(\mathbf{x})` such that :math:`-\nabla\phi(\mathbf{x})` is the right-hand side of Eq.:eq:`grad_flow_generic`. In this case,
the algorithm :eq:`update_rule` can be seen as a gradient descent algorithm towards the minimizer of the functional :math:`\phi`. As already remarked, such minimizer will be the steady-state of the PDE :eq:`grad_flow_generic`. Typically, the steady-state of
diffusion operators is a flat solution.

However, non-trivial steady-states are achieved when the diffusion operators are used for PDE-based image restoration. Indeed, let us assume that we observe a noisy version :math:`\mathbf{y}` of the true image :math:`\mathbf{x}`, so that :math:`\mathbf{y}=\mathbf{x}+\boldsymbol{\varepsilon}`,
where :math:`\boldsymbol{\varepsilon}` is the noise corrupting the image. If we know the noise distribution (Gaussian/Poisson/...), we can define a likelihood function :math:`l(\mathbf{y}\vert\mathbf{x})` describing the likelihood of observing :math:`\mathbf{y}`
if the ground truth is :math:`\mathbf{x}`. Typically, the inverse problem of recovering :math:`\mathbf{x}` from the noisy :math:`\mathbf{y}` is an ill-posed problem. *Regularization* is a popular way of addressing such ill-posedness: the image
restoration problem is formulated as the penalized optimization problem

.. math::
    \mathbf{x}^* = \arg \min_{\mathbf{x}}l(\mathbf{y}\vert\mathbf{x}) + \lambda \phi(\mathbf{x}),
   :label: penalized_opt

where :math:`\phi` is the so-called regularization functional and :math:`\lambda` the regularization parameter determining the amount of imposed regularization. To minimize :eq:`penalized_opt`, we can consider the *gradient flow*

.. math::
    \frac{\partial{\mathbf{x}}}{\partial{t}} = - \nabla l(\mathbf{y}\vert\mathbf{x}) - \lambda \nabla \phi(\mathbf{x}).
   :label: gradient_flow_penalized

Now, any diffusion operator can be used to replace the term :math:`-\nabla \phi(\mathbf{x})` in Eq.:eq:`gradient_flow_penalized` by the right-hand side of the PDE :eq:`grad_flow_generic`.
Therefore, **diffusion operators can be used for PDE-based image restoration** combining them with a suitable likelihood term. If the diffusion operators admit a potential :math:`\phi`, the penalized optimization problem :eq:`penalized_opt` can be explicitly
defined. However, diffusion operators do not necessarily derive from a potential. As a matter of fact, many of them don't. Nevertheless, they can still be used for PDE-based image smoothing based on Eq.:eq:`gradient_flow_penalized`, with the only caveat
that the resulting minimizer cannot be interpreted as the minimizer of an energy functional :math:`l(\mathbf{y}\vert\cdot)+\lambda\phi(\cdot)`.


.. toctree::
   :maxdepth: 1
   :hidden:
   :includehidden:

   api/index
   examples/index
   references