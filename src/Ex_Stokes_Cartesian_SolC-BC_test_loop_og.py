# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] magic_args="[markdown]"
# # Stokes Benchmark SolCx
# %%
# options = PETSc.Options()
# options["help"] = None


# %%
import petsc4py
from petsc4py import PETSc

# %%
import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function
import numpy as np

import sympy
from sympy import Piecewise


# %%
def plotFig(mesh, v, p, v_proj):
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 1200]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    # pvmesh.point_data["V"] = uw.function.evaluate(v.sym, mesh.data)
#     # with mesh.access(v):
#     for i in mesh.dim:
#         v_projection = uw.systems.Projection(mesh, v_proj)
#         v_projection.uw_function = v.sym[i]
#         v_projection.smoothing = 1.0e-3
#         v_projection.solve(_force_setup=True)
#         with mesh.access(v_proj):
#             pvmesh.point_data[f"V{i}"] = v_proj
            
#     vel = 
    
#         pvmesh.point_data["V"] = v_proj.dot(v_proj)

    arrow_loc = np.zeros((mesh._centroids.shape[0], 3))
    arrow_loc[:, 0:2] = mesh._centroids[...]

    arrow_length = np.zeros((mesh._centroids.shape[0], 3))
    arrow_length[:, 0] = uw.function.evaluate(v.sym[0], mesh._centroids)
    arrow_length[:, 1] = uw.function.evaluate(v.sym[1], mesh._centroids)

    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_axes()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        # scalars="V",
        use_transparency=False,
        opacity=1.0,
    )

    pl.add_arrows(arrow_loc, arrow_length, mag=3)

    pl.show(cpos="xy")


# %%
def solCx_model(n_els, penalty, weak_BC=True, viscosity=[1e6,1]):
    
    # mesh = uw.meshing.UnstructuredSimplexBox(
    #                                         minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / n_els, qdegree=2
    #                                         )
    
    mesh = uw.meshing.StructuredQuadBox(
                                        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),elementRes=(n_els, n_els)
                                        )
    
    
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    v_proj = uw.discretisation.MeshVariable("v", mesh, 1, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    
    
    x, y = mesh.CoordinateSystem.X
    
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)
    
    stokes.bodyforce = sympy.Matrix(
                                    [0, sympy.cos(sympy.pi * x) * sympy.sin(sympy.pi * y)]
                                    )
    if weak_BC == True:
        res = 1 / n_els
        hw = 1000 / res
        surface_fn = sympy.exp(-((y - 1.0) ** 2) * hw)
        base_fn = sympy.exp(-(y**2) * hw)
        right_fn = sympy.exp(-((x - 1.0) ** 2) * hw)
        left_fn = sympy.exp(-(x**2) * hw)
        
        # penalty = 100*(1/mesh.get_min_radius()**2)
        
        # penalty = 1.0e6 #325 * (1/(mesh.get_min_radius()**2))

        stokes.bodyforce[0] -= penalty * v.sym[0] * (left_fn + right_fn)
        stokes.bodyforce[1] -= penalty * v.sym[1] * (surface_fn + base_fn)
        
    else:
        stokes.add_dirichlet_bc(
            (0.0, 0.0), ["Top", "Bottom"], 1)  # top/bottom: components, function, markers
        stokes.add_dirichlet_bc(
            (0.0, 0.0), ["Left", "Right"], 0)  # left/right: components, function, markers
        

    x_c = 0.5
    
    viscosity_fn = sympy.Piecewise(
        (viscosity[0], x > x_c),
        (viscosity[1], True)               )
    
    
    stokes.constitutive_model.Parameters.viscosity = viscosity_fn
    stokes.saddle_preconditioner = 1 / stokes.constitutive_model.Parameters.viscosity


    stokes.solve(zero_init_guess=True)
    
    # if uw.mpi.size == 1:
    #     plotFig(mesh=mesh, v=v, p=p, v_proj=v_proj)
        
    volume_int = uw.maths.Integral(mesh, 1.0)
    volume = volume_int.evaluate()
    v_dot_v_int = uw.maths.Integral(mesh, stokes.u.fn.dot(stokes.u.fn))
        
    def vrms():
        import math

        v_dot_v = v_dot_v_int.evaluate()
        return math.sqrt(v_dot_v / volume)
    
    return vrms()
        
    




# %%
penalties = np.arange(1e9, 5e9+1e6, 1e9)
penalties

# %%
FSS_16 = solCx_model(n_els=16, weak_BC=False, penalty = 0)

penalties = np.arange(2.9e10, 2.95e10+1e6, 0.01e10)


penalty_vrms_16 = []


for penalty in penalties:
    penalty_vrms_16.append( solCx_model(n_els=16, weak_BC=True, penalty=penalty) )
    
# # ### closest value when n_els = 16
# # penalty = 2.91e10, ~0.0071% difference in v_rms



# %%
rel_error = []
for vrms in penalty_vrms_16:
    relative_error = ((vrms - FSS_16) / FSS_16) * 100
    rel_error.append(abs(relative_error))
    #print(relative_error)

print(np.min(rel_error))

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(dpi = 100)
ax.plot(penalties, rel_error, "o-")
#ax.set_xscale("log")

# %%
FSS_32 = solCx_model(n_els=32, weak_BC=False, penalty = 0, viscosity=[1,1e3])

# penalties = np.arange(4.5e7, 5e7+1e5, 1e6)


# penalty_vrms_32 = []

# for penalty in penalties:
#     penalty_vrms_32.append( solCx_model(n_els=32, weak_BC=True, penalty=penalty) )
    
# # ### closest value when n_els = 32   
# # penalty = 4.6e7, ~0.007% difference in v_rms

penalty = 4.6e7

FSW_32 = solCx_model(n_els=32, weak_BC=True, penalty=penalty, viscosity=[1,1e3])

relative_error = ((FSW_32 - FSS_32) / FSS_32) * 100
relative_error

# for vrms in penalty_vrms_32:
#     relative_error = ((vrms - FSS_32) / FSS_32) * 100
#     print(relative_error)

# %%
FSS_64 = solCx_model(n_els=64, weak_BC=False, penalty = 0)

penalties = np.arange(5e6, 5.5e6+1e5, 1e5)


penalty_vrms_64 = []

for penalty in penalties:
    penalty_vrms_64.append( solCx_model(n_els=64, weak_BC=True, penalty=penalty) )
    
### closest value when n_els = 64    
### penalty = 5.3e6, ~0.0012% difference in v_rms


for vrms in penalty_vrms_64:
    relative_error = ((vrms - FSS_64) / FSS_64) * 100
    print(relative_error)

# %%


penalties

# %%
# FSS_96 = solCx_model(n_els=96, weak_BC=False, penalty = 0)

penalties = np.arange(4.93e6, 4.98e6, 1e4)


penalty_vrms_96 = []

for penalty in penalties:
    penalty_vrms_96.append( solCx_model(n_els=96, weak_BC=True, penalty=penalty) )
    
### closest value when n_els = 96   
### penalty = 4.95e6, ~5.952858458274693e-05% difference in v_rms


for vrms in penalty_vrms_96:
    relative_error = ((vrms - FSS_96) / FSS_96) * 100
    print(relative_error)

# %%
n_els= 96

mesh = uw.meshing.StructuredQuadBox(
                                        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),elementRes=(n_els, n_els)
                                        )

mesh.get_min_radius()

# %%
import numpy as np

penalties = np.array([2.91e10, 4.6e7, 5.3e6, 4.95e6 ])
elementRes = np.array([16, 32, 64, 96])
mesh_min_res = np.array([0.0312499999997905, 0.015624999999893197, 0.007812499999946043, 0.005208333333297177])

import matplotlib.pyplot as plt

# plt.plot(elementRes, mesh_min_res, ls='-.', marker='o')
# plt.xlabel('element res')
# plt.show()


# plt.plot(elementRes, np.log10(penalties), ls='-.', marker='o')
# plt.xlabel('')
# plt.show()


plt.plot(mesh_min_res, np.log10(penalties), ls='-', marker='o')
plt.xlabel('mesh min res')
plt.ylabel('log$_{10}$ penalty')

penalty = (mesh_min_res**10)+6

plt.plot(mesh_min_res, penalty, ls=':')


plt.show()




# %%
x = mesh_min_res
y = penalties

# %%
np.polyfit(x, y, 3)

# %%
mesh_min_res**np.polyfit(x, y, 2)[0]

# %%
plt.plot(mesh_min_res, np.log10(penalties), ls='-', marker='o')
plt.plot(mesh_min_res, mesh_min_res**np.polyfit(x, y, 2)[0]*mesh_min_res**np.polyfit(x, y, 2)[1]+*np.polyfit(x, y, 2)[2])
plt.xlabel('mesh min res')
plt.ylabel('log$_{10}$ penalty')

# %%
mesh_min_res

# %%
for i in range(1,4):
    x = mesh_min_res
    y = np.log10(penalties)
    fit = np.polyfit(x, y, deg=i)
    
    plt.plot(x, y, ls='-.', marker='o')
    plt.plot(x, fit)
    plt.show()

# %%

# %%

