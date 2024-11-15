import numpy as np
import matplotlib.pyplot as plt
import pyxu.opt.solver as pysol
import pyxu.abc.solver as pysolver
import pyxu.opt.stop as pystop
import pyxu_diffops.operator as pyxop
import skimage as skim

# Import RGB image
image = skim.data.cat().astype(float)
print(image.shape)  # (300, 451, 3)

# Move color-stacking axis to front (needed for pyxu stacking convention)
image = np.moveaxis(image, 2, 0)
print(image.shape)  # (3, 300, 451)

# Instantiate diffusion operator
edge_enh_diffop = pyxop.AnisEdgeEnhancingDiffusionOp(dim_shape=(3, 300, 451), beta=10)

# Define PGD solver, with stopping criterion and starting point x0
stop_crit = pystop.MaxIter(n=100)

# Perform 50 gradient flow iterations starting from x0
PGD = pysol.PGD(f=edge_enh_diffop, show_progress=False, verbosity=100)
PGD.fit(**dict(mode=pysolver.SolverMode.BLOCK, x0=image, stop_crit=stop_crit,
               tau=2 / edge_enh_diffop.diff_lipschitz))
edge_enh_image = PGD.solution()

# Reshape images for plotting.
image = np.moveaxis(image, 0, 2)
edge_enh_image = np.moveaxis(edge_enh_image, 0, 2)

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image.astype(int))
ax[0].set_title("Image", fontsize=15)
ax[0].axis('off')
ax[1].imshow(edge_enh_image.astype(int))
ax[1].set_title("100 iterations Anis. Edge Enhancing", fontsize=15)
ax[1].axis('off')