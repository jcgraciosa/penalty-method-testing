# %% [markdown]
# ## Boussinesq Approximation convection in an annulus - MMS
# - Perform MMS

# %%
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import math, assess

import os

import sympy
from sympy.vector import gradient, divergence, dot
from sympy import Piecewise, And
from copy import deepcopy 
import pickle
import argparse

# %%
# TALA parameters
viscosity = 1       # iso-viscous case
viscosity_low = 1

# if using FS, can have case_num equal to 1 or 2 
# if caseNum = 1, then have low viscosity in heaven ... lol - fixed stars
# if caseNum = 2, then have low viscosity in hell ... 
caseNum = 2

#ptf = 1e6       # penalty term factor                  
# caseNum = 2 - > more ptf = 1e7 or 1e8 - only works for hwNum = 1
innerFrac = 0.3 # only valid for caseNum = 2

tol = 1e-5         # also used when using no slip 

#res = 0.03        # 0.0375# 0.075
#qdeg = 3
use_snes = True
fwhm_list = np.array([0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.007, 0.008, 0.009, 0.01, 0.02])
#fwhm_list = [0.02]
#fwhm_list = [0.001, 0.002, 0.003, 0.005, 0.007, 0.009, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1]

p_norm_arr = np.zeros(len(fwhm_list))
u_norm_arr = np.zeros(len(fwhm_list))

Tdeg = 3

# parameters for the analytical solution
n = 8  # wave number
k = 2  # polynomial order of radial density variation

### FS - free slip top, no slip base
### NS - no slip top and base
boundaryConditions = 'FS'

# not used currently 
## SM - smooth density perturbation
## DL - delta density perturbation 
# smoothOrDelta = "SM"


# %%
### set reference values

if boundaryConditions == "NS":

    outerRadius   = 6370e3
    internalRadius= (6370e3 - 660e3) ### UM - LM transition
    innerRadius   = 3480e3
    refLength     = (outerRadius - innerRadius) ### thickness of mantle

else:
    if caseNum == 1:
        atmosRadius   = 1000e3   # radius of "atmosphere" or "space"

        outerRadius   = 6370e3 + atmosRadius
        internalRadius= 6370e3
        innerRadius   = 3480e3
        refLength     = (internalRadius - innerRadius) ### thickness of mantle

    else: # only have 2 cases

        outerRadius   = 6370e3
        internalRadius= 3480e3
        innerRadius   = innerFrac*internalRadius
        refLength     = (outerRadius - internalRadius) ### thickness of mantle

rO   = outerRadius / refLength
rInt = internalRadius / refLength
rI   = innerRadius / refLength

# use these values when calculating the analytical solutions

# %%
# prepare the analytical solution 

if boundaryConditions == "NS":
    Rp, Rm = rO, rI
    solution = assess.CylindricalStokesSolutionSmoothZeroSlip(n, k, Rp=Rp, Rm=Rm)
else:
    if caseNum == 1:
        Rp, Rm = rInt, rI # fixed stars
    else:
        Rp, Rm = rO, rInt # fixed hell
        #Rp, Rm = rO, rI # fixed hell
    solution = assess.CylindricalStokesSolutionSmoothFreeSlip(n, k, Rp=Rp, Rm=Rm)

print(Rp, Rm)

parser = argparse.ArgumentParser()
parser.add_argument('-q', "--qdeg", type=int, required=True)
parser.add_argument('-p', "--ptf", type=int, required=True)
parser.add_argument('-r', "--res", type=float, required=True) # float for annulus
args = parser.parse_args()

qdeg    = args.qdeg
ptf     = args.ptf      
res     = args.res

# %%
if boundaryConditions == "NS":
    #outdir = f'/Users/jgra0019/Documents/codes/uw3-dev/TALA-EBA-benchmark/out/Annulus-MMS/BA-Annulus-MMS-{boundaryConditions}/'
    outdir = f"./out/Annulus-MMS/BA-Annulus-MMS-{boundaryConditions}/"
else:
    if caseNum == 1:
        #outdir = f'/Users/jgra0019/Documents/codes/uw3-dev/TALA-EBA-benchmark/out/Annulus-MMS/output/BA-Annulus-MMS-{boundaryConditions}-case{caseNum}/'
        outdir = f"./out/Annulus-MMS/output/BA-Annulus-MMS-{boundaryConditions}-case{caseNum}/"
    elif caseNum == 2:
        #outdir = f'/Users/jgra0019/Documents/codes/uw3-dev/TALA-EBA-benchmark/out/Annulus-MMS/output/BA-Annulus-MMS-{boundaryConditions}-case{caseNum}-innerFrac{innerFrac}/'
        outdir = f"./out/Annulus-MMS/output/BA-Annulus-MMS-{boundaryConditions}-case{caseNum}-innerFrac{innerFrac}/"

#outputPath = outdir + f"/run{args.idx}_{res}"
outputPath = outdir
print(outputPath)

# %%
if uw.mpi.rank == 0:
    # checking if the directory demo_folder 
    # exist or not.
    if not os.path.exists(outdir):

        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(outdir)

# %%

for idx, fwhm in enumerate(fwhm_list):
    print(f"fwhm: {fwhm}")
    if boundaryConditions == "NS":
        meshball = uw.meshing.AnnulusInternalBoundary(radiusOuter=rO, 
                                                    radiusInternal=rInt, 
                                                    radiusInner=rI, 
                                                    cellSize=res, 
                                                    cellSize_Outer=res, 
                                                    qdegree=qdeg)
    else:
        meshball = uw.meshing.AnnulusInternalBoundary(radiusOuter=rO, 
                                                    radiusInternal=rInt, 
                                                    radiusInner=rI, 
                                                    cellSize=res if caseNum == 1 else res, 
                                                    #cellSize=res if caseNum == 1 else 2*res, 
                                                    cellSize_Outer=res if caseNum == 1 else res, 
                                                    #cellSize_Outer=2*res if caseNum == 1 else res, 
                                                    qdegree=qdeg)

    # create the mesh variables
    v_soln      = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
    p_soln      = uw.discretisation.MeshVariable("P", meshball, 1, degree=1)
    v_soln_r    = uw.discretisation.MeshVariable("UR", meshball, 1, degree=2) # velocity R
    v_soln_th   = uw.discretisation.MeshVariable("UTH", meshball, 1, degree=2) # velocity theta

    # analytical values 
    v_ana       = uw.discretisation.MeshVariable("U2", meshball, meshball.dim, degree=2)
    p_ana       = uw.discretisation.MeshVariable("P2", meshball, 1, degree=1)
    v_ana_rth   = uw.discretisation.MeshVariable("VRTH", meshball, meshball.dim, degree=2) # cylindrical

    # density anomaly - obtained from assess
    t_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree = Tdeg) 

    # this is necessary for the penalty formulation
    meshr = uw.discretisation.MeshVariable(r"r", meshball, 1, degree=1) # mesh node radius

    # Some useful coordinate stuff
    x, y = meshball.CoordinateSystem.X
    ra, th = meshball.CoordinateSystem.xR

    # set-up calculators
    v_soln_r_calc = uw.systems.Projection(meshball, v_soln_r)
    v_soln_r_calc.uw_function = (x*v_soln.sym[0] + y*v_soln.sym[1])/sympy.sqrt(x**2 + y**2)

    v_soln_th_calc = uw.systems.Projection(meshball, v_soln_th)
    v_soln_th_calc.uw_function = (x*v_soln.sym[1] - y*v_soln.sym[0])/sympy.sqrt(x**2 + y**2)

    # piecewise version of the viscosity
    if caseNum == 1: # stars version
        viscFunc = Piecewise((viscosity_low, ra > rInt), (viscosity, True))
    else:  # hell version
        viscFunc = Piecewise((viscosity_low, ra < rInt), (viscosity, True))

    #display(viscFunc)

    # Create Stokes object
    stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes")

    ### Add constitutive model
    stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshball.dim)
    stokes.constitutive_model.Parameters.viscosity = viscFunc

    radius_fn = sympy.sqrt(meshball.X.dot(meshball.X))  # normalise by outer radius if not 1.0
    unit_rvec = meshball.X / (radius_fn)
    gravity_fn = radius_fn

    # density anomalies
    with meshball.access(t_soln):
        for i, coords in enumerate(t_soln.coords): 
            
            # convert from cartesian to cylindrical
            rc = math.sqrt(coords[0]**2 + coords[1]**2)
            pc = math.atan2(coords[1], coords[0])

            # use cartesian method
            #t_soln.data[i] = solution.delta_rho_cartesian([coords[0], coords[1]])
            if rc >= Rm and rc <= Rp:
                t_soln.data[i] = solution.delta_rho_cartesian([coords[0], coords[1]])
            else:
                t_soln.data[i] = 0 

    # analytical velocities
    with meshball.access(v_ana):
        for i, coords in enumerate(v_ana.coords):
            
            # convert from cartesian to cylindrical
            rc = math.sqrt(coords[0]**2 + coords[1]**2)
            pc = math.atan2(coords[1], coords[0])
            
            if rc >= Rm and rc <= Rp:
                v_ana.data[i, 0], v_ana.data[i, 1] = solution.velocity_cartesian([coords[0], coords[1]])
            else:
                v_ana.data[i, 0], v_ana.data[i, 1] = (0, 0)

    with meshball.access(v_ana_rth):
        for i, coords in enumerate(v_ana_rth.coords):
            # convert from cartesian to cylindrical
            rc = math.sqrt(coords[0]**2 + coords[1]**2)
            pc = math.atan2(coords[1], coords[0])
            
            if rc >= Rm and rc <= Rp:
                v_ana_rth.data[i, 0] = solution.u_r(rc, pc)
                v_ana_rth.data[i, 1] = solution.u_phi(rc, pc)
            else:
                v_ana_rth.data[i, 0] = 0
                v_ana_rth.data[i, 1] = 0

    # analytical pressure
    with meshball.access(p_ana):
        for i, coords in enumerate(p_ana.coords):
            
            rc = math.sqrt(coords[0]**2 + coords[1]**2)
            pc = math.atan2(coords[1], coords[0])

            if rc >= Rm and rc <= Rp:
                p_ana.data[i] = solution.pressure_cartesian([coords[0], coords[1]])
            else:
                p_ana.data[i] = 0
        
    with meshball.access(meshr):
        meshr.data[:, 0] = uw.function.evaluate(sympy.sqrt(x**2 + y**2), meshball.data, meshball.N)  # cf radius_fn which is 0->1

    ### set up bouyancy force
    buoyancy_force = -t_soln.sym[0]
    stokes.bodyforce = buoyancy_force * unit_rvec

    # Add boundary conditions
    if boundaryConditions == 'FS':

        # hw = hwNum / meshball.get_min_radius() #numerator originally 1e3 - original code
        #hw = hwNum*meshball.get_min_radius() # fwhm
        hw = fwhm 
        sdev = hw/2.355

        ### mark the Earth's surface nodes and CMB base nodes
        if caseNum == 1: # fixed stars
            surface_fn = sympy.exp(-((meshr.sym[0] - rInt)**2) / (2*sdev**2))
            base_fn = sympy.exp(-((meshr.sym[0] - rI)**2) / (2*sdev**2))
        else: # fixed hell 
            surface_fn = sympy.exp(-((meshr.sym[0] - rO)**2) / (2*sdev**2))
            base_fn = sympy.exp(-((meshr.sym[0] - rInt)**2) / (2*sdev**2))

        free_slip_penalty_upper = v_soln.sym.dot(unit_rvec) * unit_rvec * surface_fn
        free_slip_penalty_lower = v_soln.sym.dot(unit_rvec) * unit_rvec * base_fn
        
        ### Buoyancy force RHS plus free slip surface enforcement
        penalty_terms_upper = ptf * free_slip_penalty_upper
        penalty_terms_lower = ptf * free_slip_penalty_lower
        
        ### Free slip upper and lower only
        stokes.bodyforce -= penalty_terms_upper 
        stokes.bodyforce -= penalty_terms_lower
        
        
        # No slip outermost boundary
        if caseNum == 1:
            stokes.add_dirichlet_bc((0.0, 0.0), meshball.boundaries.Upper.name, (0, 1))
        else:
            stokes.add_dirichlet_bc((0.0, 0.0), meshball.boundaries.Lower.name, (0, 1))

        # set tolerance value
        stokes.tolerance = 1/ptf
    else:  
        stokes.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1)) # can also use meshball.boundaries.Upper.name
        stokes.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))

        stokes.tolerance                   = tol

    ### initial solve
    if uw.mpi.size == 1:
        if not use_snes: 
            stokes.petsc_options['pc_type']   = 'lu'

    stokes.petsc_options['snes_max_it']   = 500

    # check the stokes solve converges
    stokes.solve(zero_init_guess=True)

    # do conversion to cylindrical
    v_soln_r_calc.solve()
    v_soln_th_calc.solve()

    # calculate the L2 norms for the velocity and pressure
    # zero out the values in the extra domain
    if boundaryConditions == "FS":
        with meshball.access(v_soln):
            for i, coords in enumerate(v_soln.coords):
                # convert from cartesian to cylindrical
                rc = math.sqrt(coords[0]**2 + coords[1]**2)
                pc = math.atan2(coords[1], coords[0])
                
                if rc >= Rm and rc <= Rp:
                    pass
                else:
                    v_soln.data[i, 0] = 0
                    v_soln.data[i, 1] = 0

        with meshball.access(p_soln):
            for i, coords in enumerate(p_soln.coords):
                # convert from cartesian to cylindrical
                rc = math.sqrt(coords[0]**2 + coords[1]**2)
                pc = math.atan2(coords[1], coords[0])
                
                if rc >= Rm and rc <= Rp:
                    pass
                else:
                    p_soln.data[i, 0] = 0

        # why is this put in a separate place?
        with meshball.access(v_soln_r, v_soln_th):
            for i, coords in enumerate(v_soln_r.coords):
                # convert from cartesian to cylindrical
                rc = math.sqrt(coords[0]**2 + coords[1]**2)
                pc = math.atan2(coords[1], coords[0])
                
                if rc >= Rm and rc <= Rp:
                    pass
                else:
                    v_soln_r.data[i, 0] = 0
                    v_soln_th.data[i, 0] = 0

    # use cartesian representation, but this is the same as using cylindrical (this was verified)
    u_diff_exp = (v_soln.sym[0] - v_ana.sym[0])**2 + (v_soln.sym[1] - v_ana.sym[1])**2

    u_diff_norm_xy = uw.maths.Integral(meshball, u_diff_exp).evaluate()
    u_norm_xy = uw.maths.Integral(meshball,  v_ana.sym[0]**2 +  v_ana.sym[1]**2).evaluate()

    u_norm_arr[idx] = math.sqrt(u_diff_norm_xy/u_norm_xy)

    p_diff_norm_xy = uw.maths.Integral(meshball, (p_ana.sym[0] - p_soln.sym[0])**2).evaluate()
    p_norm_xy = uw.maths.Integral(meshball,  p_ana.sym[0]**2).evaluate()

    p_norm_arr[idx] = math.sqrt(p_diff_norm_xy/p_norm_xy)

    if uw.mpi.rank == 0:
        print(f"u norm: {u_norm_arr[idx]}")
        print(f"p norm: {p_norm_arr[idx]}")

    # save the current norm arrays
    with open(outdir + f"/{boundaryConditions}_case{caseNum}_qdeg{qdeg}_res{res}_ptf{ptf:.1e}_up_norm.pkl", "wb") as f:
        pickle.dump([u_norm_arr, p_norm_arr], f)


# after the loop
# save the current norm arrays
with open(outdir + f"/{boundaryConditions}_case{caseNum}_qdeg{qdeg}_res{res}_ptf{ptf:.1e}_up_norm.pkl", "wb") as f:
    pickle.dump([u_norm_arr, p_norm_arr], f)

# %%
# import pandas as pd

# if boundaryConditions == "NS":
#     fname = outputPath + f"norm_{boundaryConditions}_case{caseNum}_n{n}_k{k}.csv"
# else:
#     if use_snes:
#         fname = outputPath + f"t2_norm_snes_{boundaryConditions}_ptf{ptf:.1e}_res{res}_qdeg{qdeg}_n{n}_k{k}.csv"
#     else:
#         fname = outputPath + f"t2_norm_lu_{boundaryConditions}_ptf{ptf:.1e}_res{res}_qdeg{qdeg}_n{n}_k{k}.csv"

# df = {"FWHM": fwhm_list,
#     "U_NORM": u_norm_arr,
#     "P_NORM": p_norm_arr}

# df = pd.DataFrame(df)

# df.to_csv(fname, index = False)

# %%
print(u_norm_arr.min())
print(p_norm_arr.min())

# # %%
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(dpi = 150)
# ax.plot(fwhm_list, u_norm_arr, "-o")
# ax.set_xscale("log")
# ax.set_yscale("log")

# fig, ax = plt.subplots(dpi = 150)
# ax.plot(fwhm_list, p_norm_arr, "-o")
# ax.set_xscale("log")
# ax.set_yscale("log")

# %%
# #maybe for checking something
# if caseNum == 1:
#     test_r = np.arange(rI, rO, meshball.get_min_radius())
# else: 
#     test_r = np.arange(0, rO, meshball.get_min_radius())


# if caseNum == 1:
#     # surface_arr = np.exp(-(((test_r - rInt) / rInt) ** 2) * hw) # original code
#     # base_arr = np.exp(-(((test_r - rI) / rInt) ** 2) * hw) # original code
#     surface_arr = np.exp(-((test_r - rInt)**2) / (2*sdev**2))
#     base_arr = np.exp(-((test_r - rI)**2) / (2*sdev**2))
# else:
#     # surface_arr = np.exp(-(((test_r - rO) / rO) ** 2) * hw) # original code
#     # base_arr = np.exp(-(((test_r - rInt) / rO) ** 2) * hw) # original code
#     surface_arr = np.exp(-((test_r - rO)**2) / (2*sdev**2))
#     base_arr = np.exp(-((test_r - rInt)**2) / (2*sdev**2))

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(dpi = 100)
# ax.plot(test_r, base_arr)
# ax.plot(test_r, surface_arr)

# %%
# import matplotlib.pyplot as plt
# from matplotlib import patches

# # plot for the density perturbation 
# fig, ax = plt.subplots(dpi = 150)

# with meshball.access():
#     out = ax.scatter(t_soln.coords[:, 0], t_soln.coords[:, 1], c = t_soln.data[:], cmap = "coolwarm", s = 0.4, vmin = -1, vmax = 1)
#     fig.colorbar(out, pad = 0.01)
#     ax.set_title(r"$\delta \rho$")
#     ax.set_aspect("equal")
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_xticks([])
#     ax.set_yticks([])

#     if boundaryConditions == "FS":
#         if caseNum == 1: # fixed stars
#             circle = patches.Circle((0, 0), Rp, fill = False, linewidth = 0.1, color = "k")
#         else:
#             circle = patches.Circle((0, 0), Rm, fill = False, linewidth = 0.1, color = "k")

#         ax.add_patch(circle)

#     plt.tight_layout()
#     plt.savefig(outputPath + f"density_res{res}_n{n}_k{k}.png", dpi = "figure")

# %%
# import matplotlib.pyplot as plt
# from matplotlib import patches

# # plot for the density perturbation 
# fig, ax = plt.subplots(dpi = 100)

# with meshball.access():
#     out = ax.scatter(v_ana_rth.coords[:, 0], v_ana_rth.coords[:, 1], c = v_ana_rth.data[:, 0], cmap = "coolwarm", s = 0.4, vmin = -0.01, vmax = 0.01)
#     fig.colorbar(out, pad = 0.01)
#     ax.set_title(r"$\delta \rho$")
#     ax.set_aspect("equal")
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_xticks([])
#     ax.set_yticks([])

#     if boundaryConditions == "FS":
#         if caseNum == 1: # fixed stars
#             circle = patches.Circle((0, 0), Rp, fill = False, linewidth = 0.1, color = "k")
#         else:
#             circle = patches.Circle((0, 0), Rm, fill = False, linewidth = 0.1, color = "k")

#         ax.add_patch(circle)

#     plt.tight_layout()
#     #plt.savefig(outputPath + f"density_n{n}_k{k}.png", dpi = "figure")

# %%
# import matplotlib.pyplot as plt
# from matplotlib import patches

# # plot for the density perturbation 
# fig, ax = plt.subplots(dpi = 100)

# with meshball.access():
#     out = ax.scatter(v_soln_r.coords[:, 0], v_soln_r.coords[:, 1], c = v_soln_r.data[:], cmap = "coolwarm", s = 0.4, vmin = -0.01, vmax = 0.01)
#     fig.colorbar(out, pad = 0.01)
#     ax.set_title(r"$\delta \rho$")
#     ax.set_aspect("equal")
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_xticks([])
#     ax.set_yticks([])

#     if boundaryConditions == "FS":
#         if caseNum == 1: # fixed stars
#             circle = patches.Circle((0, 0), Rp, fill = False, linewidth = 0.1, color = "k")
#         else:
#             circle = patches.Circle((0, 0), Rm, fill = False, linewidth = 0.1, color = "k")

#         ax.add_patch(circle)

#     plt.tight_layout()
#     #plt.savefig(outputPath + f"density_n{n}_k{k}.png", dpi = "figure")

# %%
# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(3, 3, dpi = 300, figsize = (6, 5))
# fig.subplots_adjust(wspace = 0.01)
# scatter_sz = 0.05

# with meshball.access():

#     #########
#     ### row 0 - vx - analytical, numerical, difference
#     vmin = min(v_soln.data[:, 0].min(), v_ana.data[:, 0].min())
#     vmax = max(v_soln.data[:, 0].max(), v_ana.data[:, 0].max())
#     vmax = max(abs(vmin), abs(vmax))
#     vmin = -vmax

#     out = axs[0,0].scatter(v_ana.coords[:, 0], v_ana.coords[:, 1], c = v_ana.data[:, 0], cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[0,0].set_title(r"$v_x$ analytical", fontsize = 7, pad = 0.02)

#     out = axs[0,1].scatter(v_soln.coords[:, 0], v_soln.coords[:, 1], c = v_soln.data[:, 0], cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[0,1].set_title(r"$v_x$ numerical", fontsize = 7, pad = 0.02)

#     diff = v_ana.data[:, 0] - v_soln.data[:, 0]
#     vmax = max(abs(diff.min()), abs(diff.max()))
#     vmin = -vmax
#     out = axs[0,2].scatter(v_soln.coords[:, 0], v_soln.coords[:, 1], c = diff, cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[0,2].set_title(r"$\Delta v_x$", fontsize = 7, pad = 0.02)

#     #########
#     ### row 1 - vy - analytical, numerical, difference
#     vmin = min(v_soln.data[:, 1].min(), v_ana.data[:, 1].min())
#     vmax = max(v_soln.data[:, 1].max(), v_ana.data[:, 1].max())
#     vmax = max(abs(vmin), abs(vmax))
#     vmin = -vmax

#     out = axs[1,0].scatter(v_ana.coords[:, 0], v_ana.coords[:, 1], c = v_ana.data[:, 1], cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[1,0].set_title(r"$v_y$ analytical", fontsize = 7, pad = 0.02)

#     out = axs[1,1].scatter(v_soln.coords[:, 0], v_soln.coords[:, 1], c = v_soln.data[:, 1], cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[1,1].set_title(r"$v_y$ numerical", fontsize = 7, pad = 0.02)

#     diff = v_ana.data[:, 1] - v_soln.data[:, 1]
#     vmax = max(abs(diff.min()), abs(diff.max()))
#     vmin = -vmax
#     out = axs[1,2].scatter(v_soln.coords[:, 0], v_soln.coords[:, 1], c = diff, cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[1,2].set_title(r"$\Delta v_y$", fontsize = 7, pad = 0.02)

#     #########
#     ### row 2 - p - analytical, numerical, difference
#     vmin = min(p_soln.data[:].min(), p_ana.data[:].min())
#     vmax = max(p_soln.data[:].max(), p_ana.data[:].max())
#     vmax = max(abs(vmin), abs(vmax))
#     vmin = -vmax

#     out = axs[2,0].scatter(p_ana.coords[:, 0], p_ana.coords[:, 1], c = p_ana.data[:], cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[2,0].set_title(r"$p$ analytical", fontsize = 7, pad = 0.02)

#     out = axs[2,1].scatter(p_soln.coords[:, 0], p_soln.coords[:, 1], c = p_soln.data[:], cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[2,1].set_title(r"$p$ numerical", fontsize = 7, pad = 0.02)

#     diff = p_ana.data[:] - p_soln.data[:]
#     vmax = max(abs(diff.min()), abs(diff.max()))
#     vmin = -vmax
#     out = axs[2,2].scatter(p_soln.coords[:, 0], p_soln.coords[:, 1], c = diff, cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[2,2].set_title(r"$\Delta p$", fontsize = 7, pad = 0.02)

# for row in axs:
#     for ax in row:
#         ax.set_aspect("equal")
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_xticks([])
#         ax.set_yticks([])

#         if boundaryConditions == "FS":
#             if caseNum == 1: # fixed stars
#                 circle = patches.Circle((0, 0), Rp, fill = False, linewidth = 0.1, color = "k")
#             else:
#                 circle = patches.Circle((0, 0), Rm, fill = False, linewidth = 0.1, color = "k")
#             ax.add_patch(circle)

# plt.tight_layout()
# plt.savefig(outputPath + f"cartesian_results_n{n}_k{k}.png", dpi = "figure")


# %% [markdown]
# ## Plot in cylindrical components version

# %%
# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(3, 3, dpi = 150, figsize = (6, 5))
# fig.subplots_adjust(wspace = 0.01)
# scatter_sz = 0.05

# with meshball.access():

#     #########
#     ### row 0 - v_radial - analytical, numerical, difference
#     vmin = min(v_soln_r.data[:].min(), v_ana_rth.data[:, 0].min())
#     vmax = max(v_soln_r.data[:].max(), v_ana_rth.data[:, 0].max())
#     vmax = max(abs(vmin), abs(vmax))
#     vmin = -vmax

#     out = axs[0,0].scatter(v_ana_rth.coords[:, 0], v_ana_rth.coords[:, 1], c = v_ana_rth.data[:, 0], cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[0,0].set_title(r"$v_R$ analytical", fontsize = 7, pad = 0.02)

#     out = axs[0,1].scatter(v_soln_r.coords[:, 0], v_soln_r.coords[:, 1], c = v_soln_r.data[:, 0], cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[0,1].set_title(r"$v_R$ numerical", fontsize = 7, pad = 0.02)

#     diff = v_ana_rth.data[:, 0] - v_soln_r.data[:, 0]
#     vmax = max(abs(diff.min()), abs(diff.max()))
#     vmin = -vmax
#     out = axs[0,2].scatter(v_ana_rth.coords[:, 0], v_ana_rth.coords[:, 1], c = diff, cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[0,2].set_title(r"$\Delta v_R$", fontsize = 7, pad = 0.02)

#     #########
#     ### row 1 - v_theta - analytical, numerical, difference
#     vmin = min(v_soln_th.data[:].min(), v_ana_rth.data[:, 1].min())
#     vmax = max(v_soln_th.data[:].max(), v_ana_rth.data[:, 1].max())
#     vmax = max(abs(vmin), abs(vmax))
#     vmin = -vmax

#     out = axs[1,0].scatter(v_ana_rth.coords[:, 0], v_ana_rth.coords[:, 1], c = v_ana_rth.data[:, 1], cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[1,0].set_title(r"$v_{\theta}$ analytical", fontsize = 7, pad = 0.02)

#     out = axs[1,1].scatter(v_soln_th.coords[:, 0], v_soln_th.coords[:, 1], c = v_soln_th.data[:, 0], cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[1,1].set_title(r"$v_{\theta}$ numerical", fontsize = 7, pad = 0.02)

#     diff = v_ana_rth.data[:, 1] - v_soln_th.data[:, 0]
#     vmax = max(abs(diff.min()), abs(diff.max()))
#     vmin = -vmax
#     out = axs[1,2].scatter(v_ana_rth.coords[:, 0], v_ana_rth.coords[:, 1], c = diff, cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[1,2].set_title(r"$\Delta v_{\theta}$", fontsize = 7, pad = 0.02)

#     #########
#     ### row 2 - p - analytical, numerical, difference
#     vmin = min(p_soln.data[:].min(), p_ana.data[:].min())
#     vmax = max(p_soln.data[:].max(), p_ana.data[:].max())
#     vmax = max(abs(vmin), abs(vmax))
#     vmin = -vmax

#     out = axs[2,0].scatter(p_ana.coords[:, 0], p_ana.coords[:, 1], c = p_ana.data[:], cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[2,0].set_title(r"$p$ analytical", fontsize = 7, pad = 0.02)

#     out = axs[2,1].scatter(p_soln.coords[:, 0], p_soln.coords[:, 1], c = p_soln.data[:], cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[2,1].set_title(r"$p$ numerical", fontsize = 7, pad = 0.02)

#     diff = p_ana.data[:] - p_soln.data[:]
#     vmax = max(abs(diff.min()), abs(diff.max()))
#     vmin = -vmax
#     out = axs[2,2].scatter(p_soln.coords[:, 0], p_soln.coords[:, 1], c = diff, cmap = "coolwarm", s = scatter_sz, vmin = vmin, vmax = vmax)
#     cbar = fig.colorbar(out, pad = 0.02)
#     cbar.ax.tick_params(labelsize = 7)
#     axs[2,2].set_title(r"$\Delta p$", fontsize = 7, pad = 0.02)

# for row in axs:
#     for ax in row:
#         ax.set_aspect("equal")
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_xticks([])
#         ax.set_yticks([])

#         if boundaryConditions == "FS":
#             if caseNum == 1: # fixed stars
#                 circle = patches.Circle((0, 0), Rp, fill = False, linewidth = 0.1, color = "k")
#             else:
#                 circle = patches.Circle((0, 0), Rm, fill = False, linewidth = 0.1, color = "k")
#             ax.add_patch(circle)

# plt.tight_layout()#
# # plt.savefig(outputPath + f"cylindrical_results_res{res}_ptf{ptf:.1e}_qdeg{qdeg}_n{n}_k{k}.png", dpi = "figure")

# # f"t2_norm_{boundaryConditions}_ptf{ptf:.1e}_res{res}_qdeg{qdeg}_n{n}_k{k}.csv"



