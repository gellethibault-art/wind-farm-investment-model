# wind-farm-investment-model
Wind farm layout optimisation and investment analysis using PyWake, TOPFARM and Python

# Wind Farm Layout Optimisation & Investment Analysis

## Context
Academic project developed as part of the MSc Sustainable Energy Futures 
at Imperial College London. The project models a wind farm investment case 
for the Winter Hill site (UK), combining layout optimisation, AEP 
forecasting and financial analysis.

## Objectives
- Select the optimal wind turbine among three candidates based on 
  site wind conditions
- Optimise turbine layout within site boundaries to maximise Annual 
  Energy Production (AEP)
- Assess wake losses and their impact on energy yield
- Build a financial model to evaluate project bankability (NPV, IRR, 
  DSCR, downside scenarios)

## Methodology

### 1. Wind Resource Assessment
- Historical wind data processed and extrapolated to hub height 
  using a power law profile
- Weibull distribution fitted per wind direction sector (32 sectors)
- Wind rose and sector frequency analysis

### 2. Turbine Selection
Three turbines compared:
- Vestas V117-4.0 MW (IEC IB)
- Nordex N90/2500 (IEC IA)
- Senvion 6M126 (IEC IB)

Power curves, Cp and Ct coefficients modelled using BEM theory 
and tabular data.

### 3. Wake Modelling & Layout Optimisation
- Wake model: Bastankhah Gaussian (PyWake)
- Layout optimisation: TOPFARM with SciPy gradient-based driver
- Constraints: polygon boundary + minimum spacing (4D)
- Objective: maximise total AEP across 23 turbines

### 4. Financial Analysis
- Revenue modelling based on optimised AEP and electricity price 
  assumptions
- NPV / IRR analysis
- Debt sizing and DSCR computation
- Downside revenue scenarios (wind variability, price stress)

## Tools & Libraries
- Python, NumPy, Pandas, Matplotlib
- PyWake (DTU) — wake modelling
- TOPFARM (DTU) — layout optimisation
- Shapely, SciPy — geometry and statistical fitting

## Results
- Optimised layout for 23 turbines within Winter Hill boundary
- AEP per turbine computed and visualised
- Financial model outputs available in `/finance` folder

## Repository Structure
wind-farm-investment/
│
├── wind_analysis.py        # Wind resource & Weibull fitting
├── turbine_models.py       # Power curves & Ct modelling
├── layout_optimisation.py  # TOPFARM optimisation
├── financial_model.xlsx    # NPV, IRR, DSCR model
├── data/
│   └── Wind_Data.xlsx      # Raw wind data (anonymised)
└── README.md
