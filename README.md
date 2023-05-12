# PoroMechanics-Simulator

Fully implicit, multiphase poromechanical simulator, used for the examples in the Scientific Reports paper:

Uses PyParadiso for fast matrix inversions: https://github.com/haasad/PyPardisoProject

Files:

 -twoD_functions.py: contains all functions to construct the different blocks of the Jacobian matrix and Residual, as well as Gassmann's equation

 -Poromechanics2D_PistonTest.py: simple testcase

 -Poromechanics2D_HeterogenTest.py: complex testcase
 
 -RunSimul.slurm: Scheduler to run on slurm cluster
 
 -Parameters folder contains the subsurface model used for the simulations
 
 -Output folders SU_models and PM_test will contain the velocity and density models as well as the results of the simulation

The output SU_models can be used with the OpenSource package to get the seismic reflection response at the specified simulation times (https://gitlab.com/geophysicsdelft/OpenSource)
