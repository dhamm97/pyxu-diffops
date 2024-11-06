import matplotlib.pyplot as plt
import numpy as np
import pyxu.abc.solver as pysolver
import pyxu.opt.solver as pysol
import pyxu.opt.stop as pystop
import skimage as skim

import pyxu_diffops.operator as pyxop

# # Import RGB image
# image = skim.data.cat().astype(float)
# print(image.shape)  # (300, 451, 3)
#
# # Move color-stacking axis to front (needed for pyxu stacking convention)
# image = np.moveaxis(image, 2, 0)
# print(image.shape)  # (3, 300, 451)
#
# # Instantiate diffusion operator
# anis_diffop = pyxop.AnisDiffusionOp(dim_shape=(3, 300, 451), alpha=1e-3)
#
# # Define PGD solver, with stopping criterion and starting point x0
# stop_crit = pystop.MaxIter(n=100)
#
# # Perform 50 gradient flow iterations starting from x0
# PGD = pysol.PGD(f=anis_diffop, show_progress=False, verbosity=100)
# PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=image, stop_crit=stop_crit,
#                tau=2 / anis_diffop.diff_lipschitz))
# anis_image = PGD.solution()
#
# # Reshape images for plotting.
# image = np.moveaxis(image, 0, 2)
# anis_image = np.moveaxis(anis_image, 0, 2)
#
# # Plot
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(image.astype(int))
# ax[0].set_title("Image", fontsize=15)
# ax[0].axis('off')
# ax[1].imshow(anis_image.astype(int))
# ax[1].set_title("100 iterations Anisotropic smoothing", fontsize=15)
# ax[1].axis('off')
# plt.show()
#


# Define random image
image = 255 * np.random.rand(3, 300, 451)
image = skim.data.cat().astype(float)
image = np.moveaxis(image, 2, 0)

# Define vector field, diffusion process will preserve curvature along it
image_center = np.array(image.shape[1:]) / 2 + [0.25, 0.25]
curvature_preservation_field = np.zeros((2, np.prod(image.shape[1:])))
curv_pres_1 = np.zeros(image.shape[1:])
curv_pres_2 = np.zeros(image.shape[1:])
for i in range(image.shape[1]):
    for j in range(image.shape[2]):
        theta = np.arctan2(-i + image_center[0], j - image_center[1])
        curv_pres_1[i, j] = np.cos(theta)
        curv_pres_2[i, j] = np.sin(theta)
curvature_preservation_field[0, :] = curv_pres_1.reshape(1, -1)
curvature_preservation_field[1, :] = curv_pres_2.reshape(1, -1)
curvature_preservation_field = curvature_preservation_field.reshape(2, *image.shape[1:])

# Define curvature-preserving diffusion operator
CurvPresDiffusionOp = pyxop.CurvaturePreservingDiffusionOp(
    dim_shape=(3, 300, 451), curvature_preservation_field=curvature_preservation_field
)

# Perform 500 gradient flow iterations
stop_crit = pystop.MaxIter(n=500)
PGD_curve = pysol.PGD(f=CurvPresDiffusionOp, g=None, show_progress=False, verbosity=100)

PGD_curve.fit(
    **dict(
        mode=pysolver.SolverMode.BLOCK,
        x0=image,
        stop_crit=stop_crit,
        acceleration=False,
        tau=1 / CurvPresDiffusionOp.diff_lipschitz,
    )
)
curv_smooth_image = PGD_curve.solution()

# Reshape images for plotting.
image = np.moveaxis(image, 0, 2)
curv_smooth_image = np.moveaxis(curv_smooth_image, 0, 2)
# Rescale for better constrast
curv_smooth_image -= np.min(curv_smooth_image)
curv_smooth_image /= np.max(curv_smooth_image)
curv_smooth_image *= 255

# Plot
fig, ax = plt.subplots(1, 3, figsize=(20, 4))
ax[0].imshow(image.astype(int), cmap="gray", aspect="auto")
ax[0].set_title("Image")
ax[0].axis("off")
ax[1].quiver(curv_pres_2[::40, ::60], curv_pres_1[::40, ::60])
ax[1].set_title("Vector field")
ax[1].axis("off")
ax[2].imshow(curv_smooth_image.astype(int), cmap="gray", aspect="auto")
ax[2].set_title("200 iterations Curvature Preserving")
ax[2].axis("off")


plt.show()
