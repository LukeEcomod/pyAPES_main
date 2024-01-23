# pyAPES (Atmosphere-PlantEcosystem Simulator in pure Python)

### Master code, working towards publishable code & model description paper in GMD

### Recent changes (Samuli, March-June 2023):

1. revised package structure, renamed packages
1. typing, documenting interfaces
1. Sphinx-integration for online-documentation

### ToDo's:

1. Document parameters -package
1. Multi-layer model (MLM) and 2-big-leaf model versions & Notebooks as user-guides) 
1. implement soil water bucket model (from SpaFhy-style)
1. Notebooks for package-level demonstrations (user guide, case-examples for GMD)
1. debug dependencies
1. add checks on installed modules

### Future developments:

1. Sapflow and trunk water storage dynamics using 1D porous media approach
1. Energy-balance based snow scheme
1. Add 13C, 18O and 2H isotopes: for d13C and d18O see [Demo_mlm_isotopes](Demo_mlm_isotopes.ipynb)
- d13C probably requires mesophyll conductance and wood respiration to be added to pyAPES 
