# pyAPES (Atmosphere-PlantEcosystem Simulator in pure Python)

### Master code, working towards publishable code & model description paper in GMD (Tikkasalo et al. 2025)

### Recent changes:

1. revised package structure, renamed packages
2. typing, documenting interfaces
3. Sphinx-integration for online-documentation

### ToDo's:

1. implement soil water bucket model (from SpaFhy-style)
2. Notebooks for package-level demonstrations (user guide, case-examples for GMD)
3. add checks on installed modules

### Current developments:

1. Sapflow and trunk water storage dynamics using 1D porous media approach (Opa)
2. Energy-balance based snow scheme FSM (J-P)
3. Add 13C and 18O isotopes: seer isotope branch and [Demo_mlm_isotopes](Demo_mlm_isotopes.ipynb)
    - Leaf sugar pool for sunlit and shaded leafs separately doesn't make sense, because leaves are not shaded or sunlit continiously. Should be layerwise or removed completely?
