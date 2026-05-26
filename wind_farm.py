#%% ============================================================
# WIND FARM LAYOUT OPTIMISATION & INVESTMENT ANALYSIS
# Site: Winter Hill (UK)
# Author: Thibault Gellé
# MSc Sustainable Energy Futures, Imperial College London
# ============================================================

# Python dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.stats import weibull_min
from scipy.special import gamma
from scipy.optimize import fsolve
from scipy.spatial import distance_matrix
from shapely.geometry import Polygon

# PyWake dependencies
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.site import UniformWeibullSite
from py_wake import BastankhahGaussian

# TOPFARM dependencies
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm import TopFarmProblem
from topfarm.plotting import XYPlotComp
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint

plt.ion()


#%% ============================================================
# 1. WIND RESOURCE ASSESSMENT
# Load data, hub height extrapolation, Weibull fitting per sector
# ============================================================

# Load and clean wind data
df = pd.read_csv("Wind_Data.csv", usecols=[0, 1, 2, 3, 4, 5, 6], skiprows=1)
df.columns = ["day", "month", "year", "hour", "Wind_Vel_kts", "Wind_Dir", "Wind_Vel_m/s"]

# Hub height and wind speed extrapolation using power law
H = 117  # Hub height in metres (Senvion 6M126)
ws = df['Wind_Vel_m/s'].values * (H / 10) ** 0.2
wd = df['Wind_Dir'].values

# Define wind direction bins (32 sectors of 11.25°)
bins = np.arange(0, 370, 11.25)
n_sectors = len(bins) - 1
total_count = len(wd)

# Fit Weibull distribution per sector
p_wd, a_list, k_list = [], [], []

for i in range(n_sectors):
    mask = (wd >= bins[i]) & (wd < bins[i + 1])
    ws_sector = ws[mask]
    p_wd.append(len(ws_sector) / total_count)
    if len(ws_sector) > 0:
        shape, loc, scale = weibull_min.fit(ws_sector, floc=0)
        k_list.append(shape)
        a_list.append(scale)
    else:
        k_list.append(np.nan)
        a_list.append(np.nan)

p_wd = np.array(p_wd)
a_arr = np.array(a_list)
k_arr = np.array(k_list)

# Remove empty sectors and normalise frequencies
valid = ~np.isnan(a_arr)
p_wd = p_wd[valid] / np.sum(p_wd[valid])
a_arr = a_arr[valid]
k_arr = k_arr[valid]

# Create PyWake site object
site = UniformWeibullSite(
    p_wd=p_wd.tolist(),
    a=a_arr.tolist(),
    k=k_arr.tolist(),
    ti=0.14  # Turbulence intensity: 14% for IEC IB
)
print("Wind site object created successfully.")

# Plot Weibull distributions per sector
sector_centers = (bins[:-1] + bins[1:]) / 2
ws_range = np.linspace(0, np.ceil(np.max(ws)), 500)

plt.figure(figsize=(12, 8))
for i in range(len(a_arr)):
    pdf = weibull_min.pdf(ws_range, k_arr[i], scale=a_arr[i])
    avg_ws = a_arr[i] * gamma(1 + 1 / k_arr[i])
    plt.plot(ws_range, pdf,
             label=f"Sector {sector_centers[i]:.1f}° | k={k_arr[i]:.2f}, a={a_arr[i]:.2f}, avg={avg_ws:.2f} m/s")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Probability Density")
plt.title("Weibull Distributions by Wind Direction Sector")
plt.legend(fontsize=7, loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()


#%% ============================================================
# 2. TURBINE MODELLING
# Power curves, Cp and Ct for 3 candidate turbines
# BEM-based Ct correction derived from Vestas reference turbine
# ============================================================

def solve_induction(cp_val):
    """Solve for axial induction factor a from Cp using BEM theory."""
    func = lambda a: 4 * a * (1 - a) ** 2 - cp_val
    return fsolve(func, 0.3)[0]

def plot_turbine_characteristics(u, power, cp, ct, ct_est, title, ct_est_corr=None):
    """Plot power curve, Cp and Ct for a given turbine."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)

    axes[0].plot(u, power, marker='o', lw=1.8, color='#1f77b4', label='Electrical Power')
    axes[0].set(xlabel="Wind speed U [m/s]", ylabel="Power [kW]", title=f"{title} Power Curve")
    axes[0].legend()

    axes[1].plot(u, cp, marker='o', lw=1.8, color="#1fb462", label='Cp')
    axes[1].set(xlabel="Wind speed U [m/s]", ylabel="Power Coefficient Cp [-]",
                title=f"{title} Power Coefficient")
    axes[1].set_ylim(bottom=0)
    axes[1].legend()

    axes[2].plot(u, ct, marker='o', lw=1.8, color='#d62728', label='Ct')
    axes[2].plot(u, ct_est, marker='o', lw=1.8, color="#d627b3", label='Ct BEM')
    if ct_est_corr is not None:
        axes[2].plot(u, ct_est_corr, marker='o', lw=1.8, color="#d8eb46", label='Ct BEM Corrected')
    axes[2].set(xlabel="Wind speed U [m/s]", ylabel="Thrust Coefficient Ct [-]",
                title=f"{title} Thrust Coefficient")
    axes[2].set_ylim(bottom=0)
    axes[2].legend()

    plt.show()

# --- Vestas V117-4.0 MW (IEC IB) ---
u_v117 = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0,
                   9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5,
                   15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0,
                   20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0])
power_v117 = np.array([22, 78, 151, 237, 341, 466, 618, 797, 1008, 1250, 1526, 1838, 2185,
                       2551, 2915, 3253, 3555, 3807, 3950, 3990, 3998, 3999, 4000, 4000,
                       4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000,
                       4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000])
D_v117 = 117

cp_v117 = power_v117 * 1e3 / (0.5 * 1.225 * u_v117 ** 3 * np.pi / 4 * D_v117 ** 2)
ct_v117 = np.array([0.886, 0.859, 0.843, 0.834, 0.833, 0.831, 0.826, 0.824, 0.822, 0.811,
                    0.801, 0.800, 0.799, 0.782, 0.741, 0.682, 0.621, 0.562, 0.499, 0.432,
                    0.374, 0.330, 0.291, 0.260, 0.232, 0.209, 0.189, 0.172, 0.157, 0.144,
                    0.132, 0.122, 0.112, 0.104, 0.097, 0.090, 0.084, 0.080, 0.075, 0.070,
                    0.066, 0.062, 0.058, 0.055, 0.052])

a_v117 = np.array([solve_induction(cp) for cp in cp_v117])
ct_est_v117 = 4 * a_v117 * (1 - a_v117)
coeff = ct_v117 / ct_est_v117  # Correction factor derived from Vestas reference

plot_turbine_characteristics(u_v117, power_v117, cp_v117, ct_v117, ct_est_v117, "V117-4.0")

wt_1 = WindTurbine(
    name='Vestas V117-4.0MW (IEC IB)',
    diameter=D_v117,
    hub_height=H,
    powerCtFunction=PowerCtTabular(u_v117, power_v117, 'kW', ct_v117)
)

# --- Nordex N90/2500 (IEC IA) ---
u_n90 = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0,
                  9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5,
                  15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0,
                  20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0])
power_n90 = np.array([1, 37, 84, 142, 212, 294, 391, 504, 635, 785, 951, 1131, 1321,
                      1520, 1722, 1924, 2122, 2280, 2389, 2459, 2495, 2500, 2500, 2500,
                      2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500,
                      2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500])
ct_n90 = np.array([1.26, 1.15, 1.06, 1.00, 0.94, 0.89, 0.87, 0.87, 0.87, 0.86, 0.82,
                   0.78, 0.75, 0.71, 0.67, 0.64, 0.60, 0.57, 0.51, 0.43, 0.38, 0.33,
                   0.29, 0.26, 0.23, 0.21, 0.19, 0.18, 0.16, 0.15, 0.14, 0.12, 0.12,
                   0.11, 0.10, 0.09, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06])
D_n90 = 90

cp_n90 = power_n90 * 1e3 / (0.5 * 1.225 * u_n90 ** 3 * np.pi / 4 * D_n90 ** 2)
a_n90 = np.array([solve_induction(cp) for cp in cp_n90])
ct_est_n90 = 4 * a_n90 * (1 - a_n90)
ct_est_corr_n90 = ct_est_n90 * coeff

plot_turbine_characteristics(u_n90, power_n90, cp_n90, ct_n90, ct_est_n90,
                              "N90/2500", ct_est_corr=ct_est_corr_n90)

wt_2 = WindTurbine(
    name='Nordex N90/2500 (IEC IA)',
    diameter=D_n90,
    hub_height=H,
    powerCtFunction=PowerCtTabular(u_n90, power_n90, 'kW', ct_n90)
)

# --- Senvion 6M126 (IEC IB) ---
u_s6m = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
                  10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5,
                  16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0,
                  21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5,
                  27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0])
power_s6m = np.array([84, 210, 338, 506, 673, 901, 1128, 1396, 1663, 2011, 2359, 2767,
                      3175, 3717, 4258, 4666, 5074, 5415, 5756, 6050, 6150, 6150, 6150,
                      6150, 6150, 6150, 6150, 6150, 6150, 6150, 6150, 6150, 6150, 6150,
                      6150, 6150, 6150, 6150, 6150, 6150, 6150, 6150, 6150, 6150, 6150,
                      6150, 6150, 6150, 6150, 6150, 6150, 6150, 6150])
D_s6m = 126

cp_s6m = power_s6m * 1e3 / (0.5 * 1.225 * u_s6m ** 3 * np.pi / 4 * D_s6m ** 2)
a_s6m = np.array([solve_induction(cp) for cp in cp_s6m])
ct_est_s6m = 4 * a_s6m * (1 - a_s6m)
coeff_long = np.concatenate((coeff, np.ones(8)))
ct_est_corr_s6m = ct_est_s6m * coeff_long

# No manufacturer Ct available for Senvion: use BEM-corrected values
plot_turbine_characteristics(u_s6m, power_s6m, cp_s6m, ct_est_corr_s6m,
                              ct_est_s6m, "Senvion 6M126")

wt_3 = WindTurbine(
    name='Senvion 6M126 (IEC IB)',
    diameter=D_s6m,
    hub_height=H,
    powerCtFunction=PowerCtTabular(u_s6m, power_s6m, 'kW', ct_est_corr_s6m)
)


#%% ============================================================
# 3. WAKE MODELLING & LAYOUT OPTIMISATION
# Bastankhah Gaussian wake model, TOPFARM gradient-based optimisation
# Constraints: polygon boundary + minimum spacing of 4D
# ============================================================

# Build wind farm model using selected turbine (Senvion 6M126)
wf_model = BastankhahGaussian(site, wt_3)
print("Wind farm model created for Winter Hill site.")

# Site boundary coordinates (metres)
x_boundary = [73.18, 461.06, 1068.48, 1529.54, 2813.91, 2319.92, 2166.23, 1288.03]
y_boundary = [700.53, 1456.07, 2244.63, 2314.31, 1958.55, 1430.40, 799.56, 22.01]
boundary = np.transpose((x_boundary, y_boundary))
boundary_polygon = Path(np.column_stack((x_boundary, y_boundary)))

x_min, x_max = min(x_boundary), max(x_boundary)
y_min, y_max = min(y_boundary), max(y_boundary)

# Number of turbines
n_wt = 23

def sample_positions_in_polygon(n_points):
    """Sample random turbine positions within site boundary."""
    points = []
    while len(points) < n_points:
        x_rand = np.random.uniform(x_min, x_max)
        y_rand = np.random.uniform(y_min, y_max)
        if boundary_polygon.contains_point((x_rand, y_rand)):
            points.append((x_rand, y_rand))
    wt_x, wt_y = zip(*points)
    return np.array(wt_x), np.array(wt_y)

def poisson_disk_sampling(n_points, polygon, min_distance, max_attempts=1000):
    """Sample positions with minimum spacing constraint (Poisson disk)."""
    points = []
    attempts = 0
    while len(points) < n_points and attempts < max_attempts:
        x_rand = np.random.uniform(x_min, x_max)
        y_rand = np.random.uniform(y_min, y_max)
        pt = np.array([x_rand, y_rand])
        if polygon.contains_point(pt):
            if not points or np.all(np.linalg.norm(np.array(points) - pt, axis=1) >= min_distance):
                points.append(pt)
        attempts += 1
    if len(points) < n_points:
        raise ValueError("Could not place all turbines with given minimum spacing.")
    points = np.array(points)
    return points[:, 0], points[:, 1]

# Initial turbine positions
# OPTION 1: Random positions within polygon
# OPTION 2: Pre-defined positions (n_wt=12)
# OPTION 3: Poisson disk sampling with minimum spacing
OPTION = 1

if OPTION == 1:
    wt_x_initial, wt_y_initial = sample_positions_in_polygon(n_wt)

elif OPTION == 2:
    wt_x_initial = np.array([406.61, 731.90, 1038.70, 1341.81, 972.17, 1297.45,
                              1611.65, 1145.90, 1478.58, 1866.71, 2254.84, 2055.23])
    wt_y_initial = np.array([768.56, 1226.04, 1676.19, 2144.65, 753.92, 1226.04,
                              1701.81, 329.38, 779.54, 1277.27, 1917.74, 812.48])

elif OPTION == 3:
    min_spacing = 4 * D_s6m
    wt_x_initial, wt_y_initial = poisson_disk_sampling(n_wt, boundary_polygon, min_spacing)

# Run TOPFARM layout optimisation
cost_comp = PyWakeAEPCostModelComponent(
    wf_model, n_wt,
    wd=wf_model.site.default_wd,
    ws=wf_model.site.default_ws
)

tf_problem = TopFarmProblem(
    design_vars={'x': wt_x_initial.tolist(), 'y': wt_y_initial.tolist()},
    cost_comp=cost_comp,
    constraints=[
        XYBoundaryConstraint(boundary, boundary_type='polygon'),
        SpacingConstraint(min_spacing=4 * wf_model.windTurbines.diameter())
    ],
    driver=EasyScipyOptimizeDriver(maxiter=400),
    plot_comp=XYPlotComp()
)

cost, state, recorder = tf_problem.optimize()
print("Optimised turbine positions:", state)


#%% ============================================================
# 4. RESULTS & VISUALISATION
# AEP per turbine, optimised layout map
# ============================================================

x_opt = state['x']
y_opt = state['y']

# Simulate wake losses with optimised layout
sim_res = wf_model(x_opt, y_opt,
                   wd=wf_model.site.default_wd,
                   ws=wf_model.site.default_ws)

aep_per_turbine = sim_res.aep().sum('wd').sum('ws').values
total_aep = np.sum(aep_per_turbine)
print(f"Total AEP: {total_aep:.2f} GWh")
print(f"Average AEP per turbine: {np.mean(aep_per_turbine):.2f} GWh")

# Plot optimised layout with AEP per turbine
plt.figure(figsize=(9, 7))
sc = plt.scatter(x_opt, y_opt, c=aep_per_turbine, cmap='viridis', s=100, edgecolor='k', zorder=5)
plt.colorbar(sc, label='AEP per turbine [GWh]')

# Close boundary polygon for plotting
x_plot = x_boundary + [x_boundary[0]]
y_plot = y_boundary + [y_boundary[0]]
plt.plot(x_plot, y_plot, color='black', lw=1.5, label='Site boundary')

plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title(f'Optimised Wind Farm Layout — Total AEP: {total_aep:.1f} GWh')
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show(block=True)
