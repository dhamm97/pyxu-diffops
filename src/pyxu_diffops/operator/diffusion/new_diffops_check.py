import matplotlib.pyplot as plt
import numpy as np
import pyxu.abc.solver as pysolver
import pyxu.opt.solver as pysol
import pyxu.opt.stop as pystop
import skimage as skim

import pyxu_diffops.operator as diffop

# Import RGB image
image = skim.color.rgb2gray(skim.data.cat())
print(image.shape)  # (300, 451)
#
# # Instantiate diffusion operator
# pm_diffop = pyxop.PeronaMalikDiffusion(dim_shape=(1, 300, 451), beta=0.01)
#
# # Define PGD solver, with stopping criterion and starting point x0
# stop_crit = pystop.MaxIter(n=100)
#
# # Perform 50 gradient flow iterations starting from x0
# PGD = pysol.PGD(f=pm_diffop, show_progress=False, verbosity=100)
# PGD.fit(
#     **dict(
#         mode=pysolver.SolverMode.BLOCK,
#         x0=np.expand_dims(image, 0),
#         stop_crit=stop_crit,
#         tau=2 / pm_diffop.diff_lipschitz,
#     )
# )
# pm_smoothed_image = PGD.solution()
#
# # Plot
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(image, cmap="gray")
# ax[0].set_title("Image", fontsize=15)
# ax[0].axis("off")
# ax[1].imshow(pm_smoothed_image.squeeze(), cmap="gray")
# ax[1].set_title("100 iterations Perona Malik diffusion", fontsize=15)
# ax[1].axis("off")
#
# # Import RGB image
# image = skim.color.rgb2gray(skim.data.cat())
# print(image.shape)  # (300, 451)
#
# # Instantiate diffusion operator
# tikh_diffop = pyxop.TikhonovDiffusion(dim_shape=(1, 300, 451))
#
# # Define PGD solver, with stopping criterion and starting point x0
# stop_crit = pystop.MaxIter(n=100)
#
# # Perform 50 gradient flow iterations starting from x0
# PGD = pysol.PGD(f=tikh_diffop, show_progress=False, verbosity=100)
# PGD.fit(
#     **dict(
#         mode=pysolver.SolverMode.BLOCK,
#         x0=np.expand_dims(image, 0),
#         stop_crit=stop_crit,
#         tau=2 / tikh_diffop.diff_lipschitz,
#     )
# )
# tikh_smoothed_image = PGD.solution()
#
# # Plot
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(image, cmap="gray")
# ax[0].set_title("Image", fontsize=15)
# ax[0].axis("off")
# ax[1].imshow(tikh_smoothed_image.squeeze(), cmap="gray")
# ax[1].set_title("100 iterations Tikhonov smoothing", fontsize=15)
# ax[1].axis("off")
#

# # Import RGB image
# image = 100*skim.color.rgb2gray(skim.data.cat())
# dim_shape = (1, 300, 451)
# h = 1
# op = diffop.AnisCoherenceEnhancingDiffusionOp(dim_shape, alpha=1e-3, sampling=h, sigma_gd_st=2*h, smooth_sigma_st=4*h)
#
# stop_crit = pystop.MaxIter(n=100)
# # Perform 50 gradient flow iterations starting from x0
# PGD = pysol.PGD(f=op, show_progress=False, verbosity=100)
# PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=np.expand_dims(image, 0), stop_crit=stop_crit,
#                tau=2 / op.diff_lipschitz))
# smoothed_image = PGD.solution()
# # Plot
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(image, cmap="gray")
# ax[0].set_title("Image", fontsize=15)
# ax[0].axis('off')
# ax[1].imshow(smoothed_image.squeeze(), cmap="gray")
# ax[1].set_title("100 iterations Tikhonov smoothing", fontsize=15)
# ax[1].axis('off')
# plt.show()


image = skim.data.cat().astype(float)
print(image.shape)  # (300, 451, 3)
# move color-stacking axis to front (needed for pyxu stacking convention)
image = np.moveaxis(image, 2, 0)
dim_shape = (3, 300, 451)
h = 1
# op = diffop.AnisCoherenceEnhancingDiffusionOp(dim_shape, alpha=1e-3, sampling=h, sigma_gd_st=2*h, smooth_sigma_st=4*h)
# op = diffop.AnisEdgeEnhancingDiffusionOp(dim_shape, beta=10, sampling=h, sigma_gd_st=2*h, smooth_sigma_st=0*h)
# op = diffop.AnisDiffusionOp(dim_shape, alpha=1e-4)
# op = diffop.MfiDiffusion(dim_shape=image.shape, beta=100)
op = diffop.TotalVariationDiffusion(dim_shape=image.shape, beta=2)
stop_crit = pystop.MaxIter(n=100)
# Perform 50 gradient flow iterations starting from x0
PGD = pysol.PGD(f=op, show_progress=False, verbosity=100)
PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=image, stop_crit=stop_crit, tau=2 / op.diff_lipschitz))
smoothed_image = PGD.solution()
image = np.moveaxis(image, 0, 2)
smoothed_image = np.moveaxis(smoothed_image.reshape(3, 300, 451), 0, 2)
# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image.astype(int))
ax[0].set_title("Image", fontsize=15)
ax[0].axis("off")
ax[1].imshow(smoothed_image.astype(int))
ax[1].set_title("100 iterations Tikhonov smoothing", fontsize=15)
ax[1].axis("off")
plt.show()


# image = np.random.randn(300, 451)
# # Define vector field, diffusion process will preserve curvature along it
# image_center = np.array(image.shape) / 2 + [0.25, 0.25]
# curvature_preservation_field = np.zeros((2, image.size))
# curv_pres_1 = np.zeros(image.shape)
# curv_pres_2 = np.zeros(image.shape)
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         theta = np.arctan2(-i + image_center[0], j - image_center[1])
#         curv_pres_1[i, j] = np.cos(theta)
#         curv_pres_2[i, j] = np.sin(theta)
# curvature_preservation_field[0, :] = curv_pres_1.reshape(1, -1)
# curvature_preservation_field[1, :] = curv_pres_2.reshape(1, -1)
# curvature_preservation_field = curvature_preservation_field.reshape(2, *image.shape)
# # Define curvature-preserving diffusion operator
# CurvPresDiffusionOp = diffop.CurvaturePreservingDiffusionOp(dim_shape=(1, *image.shape),
#                                                             curvature_preservation_field=curvature_preservation_field)
# # Define stopping criterion and starting point
# stop_crit = pystop.MaxIter(n=500)
# # Perform 500 gradient flow iterations
# PGD_curve = pysol.PGD(f=CurvPresDiffusionOp, g=None, show_progress=False, verbosity=100)
# #PGD_curve.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=np.expand_dims(image, 0), stop_crit=stop_crit, acceleration=False,
#  #                    tau=2 / CurvPresDiffusionOp.diff_lipschitz))
# x0=255*np.random.rand(3, 1, *image.shape)
# PGD_curve.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=x0, stop_crit=stop_crit, acceleration=False,
#                      tau=2 / CurvPresDiffusionOp.diff_lipschitz))
# opt_curve = PGD_curve.solution()
# # Plot
# # fig, ax = plt.subplots(1, 3, figsize=(20, 4))
# # ax[0].imshow(image, cmap="gray", aspect="auto")
# # ax[0].set_title("Image")
# # ax[0].axis('off')
# # ax[1].quiver(curv_pres_2[::40, ::60], curv_pres_1[::40, ::60])
# # ax[1].set_title("Vector field")
# # ax[1].axis('off')
# # ax[2].imshow(opt_curve.reshape(image.shape), cmap="gray", aspect="auto")
# # ax[2].set_title("500 iterations Curvature Preserving")
# # ax[2].axis('off')
# # plt.show()
# fig, ax = plt.subplots(1, 3, figsize=(20, 4))
# ax[0].imshow(np.moveaxis(x0.squeeze(),0,2).astype(int), cmap="gray", aspect="auto")
# ax[0].set_title("Image")
# ax[0].axis('off')
# ax[1].quiver(curv_pres_2[::40, ::60], curv_pres_1[::40, ::60])
# ax[1].set_title("Vector field")
# ax[1].axis('off')
# out = np.moveaxis(opt_curve.squeeze(),0,2)
# out -= np.min(out)
# out /= np.max(out)
# out *= 255
# ax[2].imshow(out.astype(int), cmap="gray", aspect="auto")
# ax[2].set_title("500 iterations Curvature Preserving")
# ax[2].axis('off')
# plt.show()
