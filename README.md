# Imaging inverse problems
This repository contains python implementations related to the solution of inverse problems in imaging. We employ [ASTRA toolbox](https://github.com/jakobsj/astra-toolbox) to generate the projection geometry and the Radon operator.

* Inside the 'Gaussian' folder, we show a comparison between three solutions, posterior mean (analytical Gaussian-linear case), least-squares, and filtered back projection. The target example is a 2D X-ray tomography (Shepp-Logan phantom). 

  * main_CT_solutions.py is the running file
  * myradon.py implements some extra tools, in particular the code for generating the Shepp-Logan phantom

Any suggestions, corrections or improvements are kindly accepted :-)
