# import pyxu.operator.linop.diff as pydiff
# import numpy as np
#
# dim_shape = (120, 40)
#
# grad = pydiff.Gradient(dim_shape=dim_shape)
#
# arr = np.random.rand(*dim_shape)
#
# gradarr = grad(arr)
#
# print(gradarr.shape)
# #----------------
# dim_shape = (3, 120, 40)
#
# grad = pydiff.Gradient(dim_shape=dim_shape, directions=(1,2))
#
# arr = np.random.rand(*dim_shape)
#
# gradarr = grad(arr)
#
# print(gradarr.shape)
#
# # --------------
#
# arr = np.random.rand(5, *dim_shape)
#
# gradarr = grad(arr)
#
# print(gradarr.shape)
#
# gradtt = grad.T(gradarr)
#
# print(gradtt.shape)
#
# # nchannels must be one of the dim_shape entries/parameters
#
#
#
# # -----------
# dim_shape = (1, 120, 40)
#
# grad = pydiff.Gradient(dim_shape=dim_shape, directions=(1, 2))
#
# arr = np.random.rand(*dim_shape)
#
# #axis along which we might want to sum/reduce changes for stacked/unstacked case...
#
# gradarr = grad(arr)
#
# print(gradarr.shape)
#
# # -------------
#
# arr = np.random.rand(1, *dim_shape)
#
# #axis along which we might want to sum/reduce changes for stacked/unstacked case... it doesn't if we always add empty dimension though! I think we should
# # I think for us it should be: (batch, stack, channels, Nx, Ny). Like this we are always good... Ok, let's try
#
# gradarr = grad(arr)
#
# print(gradarr.shape)


import matplotlib.pyplot as plt
import numpy as np
import pyxu.abc.solver as pysolver

# import src.pyxu_diffops.operator as diffop
import pyxu.operator as diffop
import pyxu.opt.solver as pysol
import pyxu.opt.stop as pystop
import skimage as skim

# image = skim.color.rgb2gray(skim.data.cat())
# print(image.shape)  # (300, 451)
# # Instantiate diffusion operator
# pm_diffop = diffop.PeronaMalikDiffusion(dim_shape=(1, 300, 451), beta=0.01)
# # Define PGD solver, with stopping criterion and starting point x0
# stop_crit = pystop.MaxIter(n=100)
# # Perform 50 gradient flow iterations starting from x0
# PGD = pysol.PGD(f=pm_diffop, show_progress=True, verbosity=100)
# PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=np.expand_dims(image, 0), stop_crit=stop_crit,
#                tau=2 / pm_diffop.diff_lipschitz))
# pm_smoothed_image = PGD.solution()
# # Plot
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(image, cmap="gray")
# ax[0].set_title("Image", fontsize=15)
# ax[0].axis('off')
# ax[1].imshow(pm_smoothed_image.squeeze(), cmap="gray")
# ax[1].set_title("100 iterations Perona Malik diffusion", fontsize=15)
# ax[1].axis('off')
# plt.show()
#
# # Import RGB image
# image = skim.data.cat().astype(float)
# print(image.shape) #(300, 451, 3)
# # move color-stacking axis to front (needed for pyxu stacking convention)
# image = np.moveaxis(image, 2, 0)
# print(image.shape) #(3, 300, 451)
# # Instantiate diffusion operator
# pm_diffop = diffop.PeronaMalikDiffusion(dim_shape=(3, 300, 451), beta=5)
# # Define PGD solver, with stopping criterion and starting point x0
# stop_crit = pystop.MaxIter(n=100)
# # Perform 50 gradient flow iterations starting from x0
# PGD = pysol.PGD(f = pm_diffop, show_progress=True, verbosity=100)
# PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=image, stop_crit=stop_crit,
#                tau=2/pm_diffop.diff_lipschitz))
# pm_smoothed_image = PGD.solution()
# # Reshape images for plotting.
# image = np.moveaxis(image, 0, 2)
# pm_smoothed_image = np.moveaxis(pm_smoothed_image, 0, 2)
# # Plot
# fig, ax = plt.subplots(1,2,figsize=(10,5))
# ax[0].imshow(image.astype(int))
# ax[0].set_title("Image", fontsize=15)
# ax[0].axis('off')
# ax[1].imshow(pm_smoothed_image.astype(int))
# ax[1].set_title("100 iterations Perona Malik diffusion", fontsize=15)
# ax[1].axis('off')
# plt.show()

# Import RGB image
image = skim.data.cat().astype(float)
print(image.shape)  # (300, 451, 3)
# move color-stacking axis to front (needed for pyxu stacking convention)
image = np.moveaxis(image, 2, 0)
print(image.shape)  # (3, 300, 451)
# Instantiate diffusion operator
tikh_diffop = diffop.TikhonovDiffusion(dim_shape=(3, 300, 451))
# Define PGD solver, with stopping criterion and starting point x0
stop_crit = pystop.MaxIter(n=100)
# Perform 50 gradient flow iterations starting from x0
PGD = pysol.PGD(f=tikh_diffop, show_progress=False, verbosity=100)
PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=image, stop_crit=stop_crit, tau=2 / tikh_diffop.diff_lipschitz))
tikh_smoothed_image = PGD.solution()
# Reshape images for plotting.
image = np.moveaxis(image, 0, 2)
tikh_smoothed_image = np.moveaxis(tikh_smoothed_image, 0, 2)
# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image.astype(int))
ax[0].set_title("Image", fontsize=15)
ax[0].axis("off")
ax[1].imshow(tikh_smoothed_image.astype(int))
ax[1].set_title("100 iterations Tikhonov smoothing", fontsize=15)
ax[1].axis("off")
plt.show()
#
# Import RGB image
image = skim.color.rgb2gray(skim.data.cat())
print(image.shape)  # (300, 451)
# Instantiate diffusion operator
tikh_diffop = diffop.TikhonovDiffusion(dim_shape=(1, 300, 451))
# Define PGD solver, with stopping criterion and starting point x0
stop_crit = pystop.MaxIter(n=100)
# Perform 50 gradient flow iterations starting from x0
PGD = pysol.PGD(f=tikh_diffop, show_progress=False, verbosity=100)
PGD.fit(
    **dict(
        mode=pysolver.SolverMode.BLOCK,
        x0=np.expand_dims(image, 0),
        stop_crit=stop_crit,
        tau=2 / tikh_diffop.diff_lipschitz,
    )
)
tikh_smoothed_image = PGD.solution()
# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap="gray")
ax[0].set_title("Image", fontsize=15)
ax[0].axis("off")
ax[1].imshow(tikh_smoothed_image.squeeze(), cmap="gray")
ax[1].set_title("100 iterations Tikhonov smoothing", fontsize=15)
ax[1].axis("off")
plt.show()

# Import RGB image
image = skim.color.rgb2gray(skim.data.cat())
print(image.shape)  # (300, 451)
# Instantiate diffusion operator
tv_diffop = diffop.TotalVariationDiffusion(dim_shape=(1, 300, 451), beta=0.005)
# Define PGD solver, with stopping criterion and starting point x0
stop_crit = pystop.MaxIter(n=100)
# Perform 50 gradient flow iterations starting from x0
PGD = pysol.PGD(f=tv_diffop, show_progress=False, verbosity=100)
PGD.fit(
    **dict(
        mode=pysolver.SolverMode.BLOCK,
        x0=np.expand_dims(image, 0),
        stop_crit=stop_crit,
        tau=2 / tv_diffop.diff_lipschitz,
    )
)
tv_smoothed_image = PGD.solution()
# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap="gray")
ax[0].set_title("Image", fontsize=15)
ax[0].axis("off")
ax[1].imshow(tv_smoothed_image.squeeze(), cmap="gray")
ax[1].set_title("100 iterations Total Variation smoothing", fontsize=15)
ax[1].axis("off")
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import pyxu.operator.linop.diff as pydiff
# import pyxu_diffops.operator.diffusion._diffusivity as pydiffusion
# import skimage as skim
# import pyxu.operator as pyxop
#
# # Import image
# image = skim.color.rgb2gray(skim.data.cat())
# # Define Tikhonov diffusivity
# Tikhonov_diffusivity = pydiffusion.TikhonovDiffusivity(dim_shape=(1, *image.shape))
# # Evaluate diffusivity at image
# Tikhodiff_eval = Tikhonov_diffusivity(np.expand_dims(image, 0))
# # Plot
# fig, ax = plt.subplots(1, 3, figsize=(25, 5))
# p0 = ax[0].imshow(image, cmap="gray", aspect="auto")
# ax[0].set_title("Image", fontsize=15, pad=10)
# ax[0].set_title("Image", fontsize=15, pad=10)
# ax[0].axis("off")
# plt.colorbar(p0, ax=ax[0], fraction=0.04, pad=0.01)
# x = np.linspace(0, 1, 100)
# ax[1].plot(x, np.ones(x.size))
# ax[1].set_xlabel(r"$f$", fontsize=15)
# ax[1].set_ylabel(r"$(g$", fontsize=15, rotation=0, labelpad=7)
# ax[1].set_xlim([0, 1])
# ax[1].set_title("Tikhonov diffusivity function", fontsize=15, pad=10)
# p2 = ax[2].imshow(Tikhodiff_eval.squeeze(), cmap="gray", aspect="auto")
# ax[2].set_title("Tikhonov diffusivity evaluated at image", fontsize=15, pad=10)
# ax[2].axis("off")
# plt.colorbar(p2, ax=ax[2], fraction=0.04)
# plt.show()
#
# image = skim.color.rgb2gray(skim.data.cat())
# # Instantiate gaussian gradient operator
# # Define MFI diffusivity
# MFI_diffusivity = pydiffusion.MfiDiffusivity(dim_shape=(1, *image.shape), tame=True)
# # Evaluate diffusivity at image
# MFIdiff_eval = MFI_diffusivity(np.expand_dims(image, 0))
# # Plot
# fig, ax = plt.subplots(1, 3, figsize=(25, 5))
# p0 = ax[0].imshow(image, cmap="gray", aspect="auto")
# ax[0].set_title("Image", fontsize=15, pad=10)
# ax[0].axis("off")
# plt.colorbar(p0, ax=ax[0], fraction=0.04, pad=0.01)
# x = np.linspace(0, 1, 100)
# ax[1].plot(x, 1 / (1 + x))
# ax[1].set_xlabel(r"$f$", fontsize=15)
# ax[1].set_ylabel(r"$g$", fontsize=15, rotation=0, labelpad=7)
# ax[1].set_xlim([0, 1])
# ax[1].set_title("Tame MFI diffusivity function", fontsize=15, pad=10)
# p2 = ax[2].imshow(MFIdiff_eval.squeeze(), cmap="gray", aspect="auto")
# ax[2].set_title("MFI diffusivity evaluated at image", fontsize=15, pad=10)
# ax[2].axis("off")
# plt.colorbar(p2, ax=ax[2], fraction=0.04)
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# import pyxu.operator.linop.diff as pydiff
# import src.pyxu_diffops.operator as diffop
# import skimage as skim
#
# image = skim.color.rgb2gray(skim.data.cat())
# # Instantiate gaussian gradient operator
# gauss_grad = pydiff.Gradient(dim_shape=(1, *image.shape), directions=(1, 2), diff_method="gd", sigma=2)
# # Set contrast parameter beta (heuristic)
# gauss_grad_norm = np.linalg.norm(gauss_grad(np.expand_dims(image, 0)), axis=0)
# beta = np.quantile(gauss_grad_norm.flatten(), 0.9)
# # Define Perona-Malik diffusivity
# PeronaMalik_diffusivity = pydiffusion.PeronaMalikDiffusivity(
#     dim_shape=(1, *image.shape), gradient=gauss_grad, beta=beta, pm_fct="exponential"
# )
# # Evaluate diffusivity at image
# PMdiff_eval = PeronaMalik_diffusivity(np.expand_dims(image, 0))
# # Plot
# fig, ax = plt.subplots(1, 3, figsize=(25, 5))
# ax[0].imshow(image, cmap="gray", aspect="auto")
# ax[0].set_title("Image", fontsize=15, pad=10)
# ax[0].axis("off")
# x = np.linspace(0, 0.25, 100)
# ax[1].plot(x, np.exp(-(x**2) / (beta**2)))
# ax[1].set_xlabel(r"$\vert \nabla_\sigma f \vert$", fontsize=15)
# ax[1].set_ylabel(r"$g$", fontsize=15, rotation=0, labelpad=10)
# ax[1].set_xlim([0, 0.25])
# ax[1].set_title("Exponential Perona-Malik diffusivity function", fontsize=15, pad=10)
# p = ax[2].imshow(PMdiff_eval.squeeze(), cmap="gray", aspect="auto")
# ax[2].set_title("Perona-Malik diffusivity evaluated at image", fontsize=15, pad=10)
# ax[2].axis("off")
# plt.colorbar(p, ax=ax[2], fraction=0.04)
# plt.show()
#
# image = skim.color.rgb2gray(skim.data.cat())
# # Instantiate gaussian gradient operator
# grad = pydiff.Gradient(
#     dim_shape=(1, *image.shape), directions=(1, 2), diff_method="fd", scheme="forward", mode="symmetric"
# )
# # Set contrast parameter beta (heuristic)
# grad_norm = np.linalg.norm(grad(np.expand_dims(image, 0)), axis=0)
# beta = np.quantile(grad_norm.flatten(), 0.9)
# # Define Total Variation diffusivity
# TotalVariation_diffusivity = pydiffusion.TotalVariationDiffusivity(
#     dim_shape=(1, *image.shape), gradient=grad, beta=beta, tame=True
# )
# image = skim.color.rgb2gray(skim.data.cat())
# # Instantiate gaussian gradient operator
# grad = pydiff.Gradient(
#     dim_shape=(1, *image.shape), directions=(1, 2), diff_method="fd", scheme="forward", mode="symmetric"
# )
# # Set contrast parameter beta (heuristic)
# grad_norm = np.linalg.norm(grad(np.expand_dims(image, 0)), axis=0)
# beta = np.quantile(grad_norm.flatten(), 0.9)
# # Define Total Variation diffusivity
# TotalVariation_diffusivity = pydiffusion.TotalVariationDiffusivity(
#     dim_shape=(1, *image.shape), gradient=grad, beta=beta, tame=True
# )
# # Evaluate diffusivity at image
# TVdiff_eval = TotalVariation_diffusivity(np.expand_dims(image, 0))
# # Plot
# fig, ax = plt.subplots(1, 3, figsize=(25, 5))
# ax[0].imshow(image, cmap="gray", aspect="auto")
# ax[0].set_title("Image", fontsize=15, pad=10)
# ax[0].axis("off")
# x = np.linspace(0, 0.25, 100)
# ax[1].plot(x, 1 / np.sqrt(1 + x**2 / (beta**2)))
# ax[1].set_xlabel(r"$\vert \nabla_\sigma f \vert$", fontsize=15)
# ax[1].set_ylabel(r"$g$", fontsize=15, rotation=0, labelpad=10)
# ax[1].set_xlim([0, 0.25])
# ax[1].set_title("Total Variation diffusivity", fontsize=15, pad=10)
# p = ax[2].imshow(TVdiff_eval.squeeze(), cmap="gray", aspect="auto")
# ax[2].set_title("Total Variation diffusivity evaluated at image", fontsize=15, pad=10)
# ax[2].axis("off")
# plt.colorbar(p, ax=ax[2], fraction=0.04)
# plt.show()
