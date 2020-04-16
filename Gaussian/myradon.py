# ========================================================================
# Created by:
# Felipe Uribe @ DTU compute
# ========================================================================
# Adapted from 'PARALLELTOMO', 'fbp' and 'phantomgallery' (AIRToolsII)
# https://github.com/jakobsj/AIRToolsII
# ========================================================================
# Version 2020-03
# ========================================================================
import numpy as np
from scipy import sparse #, fftpack
# import itertools
# from numba import njit
# @njit



#=========================================================================
#=========================================================================
#=========================================================================
def discrete_radon_parallel(N, r, p):
    #=====================================================================
    # N    : the resolution of the image
    # r    : number of detector elements per angle
    # theta: array with the projection angles [rad]
    #=====================================================================
    theta = (np.arange(0, p)/p)*np.pi   # angular range [rad]
    d     = r-1

    # initialize vectors that contains the row numbers, the column numbers
    rows   = np.zeros(2*N*p*r, dtype=int)
    cols   = np.copy(rows)
    vals   = np.empty(2*N*p*r)
    idxend = 0

    # equally-spaced discretization: detector locations
    x0 = np.linspace(-d/2, d/2, r)
    y0 = np.zeros(r)

    # intersection lines: pixel grid
    x = np.arange(-N/2, N/2+1, 1)
    y = np.copy(x)

    # starting points for the angles
    tt, xx0, yy0 = theta.reshape((p,1)), x0.reshape((1,r)), y0.reshape((1,r))
    x0_theta_all = np.kron(np.cos(tt), xx0) - np.kron(np.sin(tt), yy0)
    y0_theta_all = np.kron(np.sin(tt), xx0) + np.kron(np.cos(tt), yy0)

    # direction vector for all rays corresponding to the angles
    aa, bb = -np.sin(theta), np.cos(theta)

    for i in range(p):
        # direction vector for the ith angle
        a, b = aa[i], bb[i]
        #
        for j in range(r):
            # get the y-coordinates of intersections with x = constant
            tx = (x - x0_theta_all[i,j])/a
            yx = b*tx + y0_theta_all[i,j]
        
            # get the x-coordinates of intersections with y = constant
            ty = (y - y0_theta_all[i,j])/b
            xy = a*ty + x0_theta_all[i,j]

            # collect the intersection times and coordinates
            t   = np.hstack((tx, ty))
            xxy = np.hstack((x, xy))
            yxy = np.hstack((yx, y))

            # sort the coordinates according to intersection time
            idd = np.argsort(t, kind='mergesort')
            xxy = xxy[idd]
            yxy = yxy[idd]

            # skip the points outside the box
            idd = (-N/2 <= xxy) & (xxy <= N/2) & (-N/2 <= yxy) & (yxy <= N/2)
            xxy = xxy[idd]
            yxy = yxy[idd]

            # skip double points
            idd = (abs(np.diff(xxy)) <= 1e-10) & (abs(np.diff(yxy)) <= 1e-10)
            idd = np.hstack([idd, False])      # to complete dim
            xxy = np.delete(xxy, np.ix_(idd))
            yxy = np.delete(yxy, np.ix_(idd))
                
            # length within cell and determines the number of hit cells 
            aval = np.sqrt(np.diff(xxy)**2 + np.diff(yxy)**2)
            col  = np.array([], dtype=int)

            if aval.size > 0:
                # if the ray is on the boundary of the box in the top or to the
                # right the ray does not by definition lie with in a valid cell
                if not (b == 0) & (abs(y0_theta_all[i,j] - N/2) < 1e-15) | \
                       (a == 0) & (abs(x0_theta_all[i,j] - N/2) < 1e-15):
                    # midpoints of the line within the cells
                    xm = 0.5*(xxy[:-1]+xxy[1:]) + N/2
                    ym = 0.5*(yxy[:-1]+yxy[1:]) + N/2

                    # translate the midpoint coordinates to index
                    col = ((np.floor(xm)*N) + (N - np.floor(ym)) - 1)#.astype('i8') 

            if len(col) != 0:
                # create the indices to store the values to vector 
                idxstart = idxend
                idxend   = idxstart + col.size 
                idx      = np.arange(idxstart, idxend)

                # store row numbers, column numbers and values
                rows[idx] = i*r + j
                cols[idx] = col
                vals[idx] = aval

    # truncate excess zeros
    rows = rows[:idxend]
    cols = cols[:idxend]
    vals = vals[:idxend]
    
    # create sparse discrete Radon operator
    R = sparse.coo_matrix((vals, (rows, cols)), shape=(p*r, N**2)).tocsc()

    return R, theta



#=========================================================================
#=========================================================================
#=========================================================================
def fbp(R, b, theta):
    #=====================================================================
    # R     : discrete Radon operator
    # b     : noisy sinogram data
    # theta : array with the projection angles [rad]
    #=====================================================================
    nextpow2 = lambda x: np.ceil(np.log2(abs(x)))
    # 
    dim = R.shape[0]
    p   = len(theta)
    r   = int(dim/p)

    # set ramp filter and frequencies
    order = max(64, 2**nextpow2(2*r))
    n     = np.arange(0, order/2+1)
    f     = n/max(n)
    f[0]  = 0.4/order
    f[-1] = 1 - f[0]
    # w     = 2*np.pi * n/order

    # zero paddding of sinogram
    f   = np.hstack([f , f[-2:0:-1]])
    aux = np.zeros((len(f)-r, p))
    b   = np.vstack([b, aux])
    
    # frequency domain filtering of each column of b
    b = np.fft.fft(b, axis=0)#fftpack.fft(b, axis=0)
    for i in range(p):
        b[:,i] *= f    
    b = np.fft.ifft(b, axis=0).real#fftpack.irfft(b)
    b = b[:r,:]

    # backprojection
    x  = (R.T @ b.flatten(order='F')) * (np.pi/(2*p))

    return x



#=========================================================================
#=========================================================================
#=========================================================================
def phantom(N):
    #=====================================================================
    # N : resolution
    #=====================================================================  
    #                  A      a      b     x0      y0    phi
    e = np.array( [ [  1,    .69,   .92,    0,       0,   0 ], 
                    [-.8,  .6624, .8740,    0,  -.0184,   0 ],
                    [-.2,  .1100, .3100,  .22,       0,  -18],
                    [-.2,  .1600, .4100, -.22,       0,   18],
                    [ .1,  .2100, .2500,    0,     .35,   0 ],
                    [ .1,  .0460, .0460,    0,      .1,   0 ],
                    [ .1,  .0460, .0460,    0,     -.1,   0 ],
                    [ .1,  .0460, .0230, -.08,   -.605,   0 ],
                    [ .1,  .0230, .0230,    0,   -.606,   0 ],
                    [ .1,  .0230, .0460,  .06,   -.605,   0 ], ] )
    #
    xn = ((np.arange(0,N)-(N-1)/2) / ((N-1)/2))#.reshape((1,N))
    Xn = np.tile(xn, (N,1))
    Yn = np.rot90(Xn)
    X  = np.zeros((N,N))

    # for each ellipse to be added     
    nn = e.shape[0]
    for i in range(nn):
        A   = e[i,0]
        a2  = e[i,1]**2
        b2  = e[i,2]**2
        x0  = e[i,3]
        y0  = e[i,4]
        phi = e[i,5]*np.pi/180        
        #
        x   = Xn-x0
        y   = Yn-y0
        idd = ((x*np.cos(phi) + y*np.sin(phi))**2)/a2 + ((y*np.cos(phi) - x*np.sin(phi))**2)/b2
        idx = np.where( idd <= 1 )

        # add the amplitude of the ellipse
        X[idx] += A
    #
    idx    = np.where( X < 0 )
    X[idx] = 0

    return X