# =============================================================================
# Created by:
# Felipe Uribe @ DTU
# =============================================================================
# conda install -c astra-toolbox/label/dev astra-toolbox
# =============================================================================
# Version 2020-03
# =============================================================================
import time
import numpy as np
import scipy as sp
import pickle
import astra
#
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True

# myfuns
from myradon import phantom

# =============================================================================
# parameters for the Radon transform
# =============================================================================
n         = 64                      # choose 64 or 100, grid size n x n (pixels)
p         = 200                     # number of projections
r         = 100                     # det_count: number of detectors
det_width = 2                       # detector size
dim       = n**2                    # dimension

# compute discrete Radon operator
angles    = np.linspace(0, np.pi, p, False)
vol_geom  = astra.create_vol_geom(n,n)
proj_geom = astra.create_proj_geom('parallel', det_width, r, angles)
proj_id   = astra.create_projector('strip', proj_geom, vol_geom) # strip, linear, line

# imitates a projection matrix with a given projection
# A = astra.OpTomo(proj_id)   
# use A for fast forward simulation: Ax = A @ x 

# generate 'true' phantom
x_true = phantom(n)

# =============================================================================
# data and noise: Gaussian likelihood
with open('data_n{}.pkl'.format(n), 'rb') as f:  # load noisy data
    b, b_data, sigma_obs = pickle.load(f)
# 
m         = b.size 
mu_obs    = np.zeros(m)         
lambd_obs = 1/(sigma_obs**2)

# =============================================================================
# analytical posterior Gaussian-linear solution
# =============================================================================
# prior
sigma_pr  = 0.03  
lambd_pr  = 1/(sigma_pr**2)
mu_pr     = np.zeros(dim)
Lambda_pr = lambd_pr * sp.sparse.identity(dim, format='csc')  # prior precision

# get Radon matrix
mat_id = astra.projector.matrix(proj_id)
R      = astra.matrix.get(mat_id)         # Radon matrix
H      = lambd_obs * (R.T @ R)            # Hessian of the negative log-likelihood 

# posterior covariance
print('\nComputing posterior covariance...')
ti = time.time()
Sigma_pos = sp.sparse.linalg.inv(H + Lambda_pr)
sigma_pos = np.sqrt(sp.sparse.csr_matrix.diagonal(Sigma_pos))
tf = time.time() - ti
print('\nElapsed time:',tf,'\n')

# posterior mean
mu_pos = Sigma_pos @ (lambd_obs * (R.T @ b_data))

# reshape to image
x_mu_pos    = mu_pos.reshape((n,n), order='F')
x_sigma_pos = sigma_pos.reshape((n,n), order='F')

#=====================================================================
# least-squares solution
#=====================================================================
x_lsq = sp.sparse.linalg.lsqr(R, b.flatten(order='F'), iter_lim=100)[0]
x_lsq = x_lsq.reshape((n,n), order='F')

#=====================================================================
# filtered back-projection solution from ASTRA
#=====================================================================
b_id   = astra.data2d.create('-sino', proj_geom, b.T)   # data object for the noisy data
rec_id = astra.data2d.create('-vol', vol_geom)          # data object for the reconstruction

# parameters for reconstruction via back-projection
cfg = astra.astra_dict('FBP')
cfg['ProjectorId']          = proj_id
cfg['ProjectionDataId']     = b_id
cfg['ReconstructionDataId'] = rec_id

# algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

# FBP
astra.algorithm.run(alg_id)
x_fbp = astra.data2d.get(rec_id).T

#=====================================================================
# plot
#=====================================================================
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
ax1.imshow(x_mu_pos, extent=[0, 1, 0, 1], aspect='equal', cmap='Blues_r')
ax1.set_title('Image ($\mu_{\mathrm{pos}}$)')
ax1.tick_params(axis='both', which='both', length=0)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax1.get_yticklabels(), visible=False)
#
ax2.imshow(x_fbp, extent=[0, 1, 0, 1], aspect='equal', cmap='Blues_r')
ax2.set_title('Image (FBP)')
ax2.tick_params(axis='both', which='both', length=0)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
#
ax3.imshow(x_lsq, extent=[0, 1, 0, 1], aspect='equal', cmap='Blues_r')
ax3.set_title('Image (LSQ)')
ax3.tick_params(axis='both', which='both', length=0)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
#
ax4.imshow(x_true, extent=[0, 1, 0, 1], aspect='equal', cmap='Blues_r')
ax4.set_title('Image (true)')
ax4.tick_params(axis='both', which='both', length=0)
plt.setp(ax4.get_xticklabels(), visible=False)
plt.setp(ax4.get_yticklabels(), visible=False)
#
plt.tight_layout()
plt.show()