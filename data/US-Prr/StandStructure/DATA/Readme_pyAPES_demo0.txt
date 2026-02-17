Samuli Launiainen Jan 15, 2025

Demo_0_ecosystem_structure_US-Prr.ipyng:

Creates vertical leaf-area density distrbutions for small (<=3m) and large (>3m) Black spruces. These two size cohorts are used to create 
two PlantTypes for simulating coupled carbon-water-energy exchange at US-Prr in Demo_1_multi_layer_model_at_USPrr

1) reads leaf-area density profiles of measured sample trees:
ladfile = r'data\\US-Prr\\StandStructure\\DATA\\Tree_allometry\\Leafarea_profile.csv'

2) Groups trees in 'small' and 'large' (or any height interval specified by user) and computes average lad(z) profiles.

3) Reads tree census data: size distribution for the footprint
censusfile = r'data\\US-Prr\\StandStructure\\DATA\\Tree_census\\TreeCensus.csv'

4) Computes number of 'small' and 'large' spruces (or each tree height bin) in the footprint area, and then computes footprint-average lad for each tree size bin