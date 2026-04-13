# pyAPES (Atmosphere-PlantEcosystem Simulator in pure Python)

### Master code, working towards publishable code & model description paper in GMD (Tikkasalo et al. 2026)

## Installation

Python 3.10 or later is required.

### pip / uv

```bash
git clone https://github.com/LukeEcomod/pyAPES_main.git
cd pyAPES_main

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

With **uv**:

```bash
uv venv && uv pip install -r requirements.txt
```

### conda / mamba

```bash
git clone https://github.com/LukeEcomod/pyAPES_main.git
cd pyAPES_main

conda env create -f environment.yml
conda activate pyapes
```

### Running the example notebooks

After installation, register the environment as a Jupyter kernel and start JupyterLab:

```bash
python -m ipykernel install --user --name pyapes --display-name "pyAPES"
jupyter lab
```

In JupyterLab, select the **pyAPES** kernel when opening a notebook (`Kernel → Change Kernel → pyAPES`). This ensures the notebook uses the correct environment with all installed packages.

The `Examples/` directory contains notebooks that demonstrate individual model components and serve as a starting point for new users.

### Recent changes:

1. revised package structure, renamed packages
1. typing, documenting interfaces
1. Sphinx-integration for online-documentation

### ToDo's:

1. implement soil water bucket model (from SpaFhy-style)
1. Notebooks for package-level demonstrations (user guide, case-examples for GMD)
1. add checks on installed modules

### Current developments:

1. Sapflow and trunk water storage dynamics using 1D porous media approach (Opa)
1. Energy-balance based snow scheme FSM (J-P)
1. 13C, 18O and 3H isotopes (Kersti)

