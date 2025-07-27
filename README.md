# openEMS-PY-Clone
Transformation of the java based openEMS genetic optimizationa algorithm into a python based application

## Installation

Use `pip` to install the project dependencies:

```bash
pip install -r requirements.txt
```

This installs packages such as `numpy`, `pandas`, `scipy`, `deap`, `matplotlib`, `tqdm`, and `pvlib` which are required for running the simulations.

## For Running it in a virtuel environment in Spyder

1. Open the Anaconda command window
2. Manouver to the directory you saved this repository: 
```bash
cd C:\Users\.......\openEMS-PY-Clone
```
4. Create the environement:
```bash
conda create -n openems-env python=3.10
```
6. Activate the environement:
```bash
conda activate openems-env
```
8. Install the packages:
```bash
conda install numpy pandas matplotlib scipy pvlib tqdm deap spyder
```
10. Open spyder by typing and entering:
```bash
spyder
```
12. In spyder open above the green "run" button: Execute -> Configuration per file and under command line option pass the variable "--days 5" setting the simulation horizon. If not set 1 year is simulated
