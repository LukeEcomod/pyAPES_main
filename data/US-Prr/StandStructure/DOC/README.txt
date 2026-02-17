Data title: Ecosystem structure data sets of an open canopy black spruce forest in Interior Alaska for ecosystem modeling and validation

Author: Hideki Kobayashi, Japan Agency for Marine-Earth Science and Technology (hkoba@jamstec.go.jp)

Date: April 18, 2023

Data description:
Data are stored in the four different directories named  “Tree_census”, “Tree_allometry”, “OverstoryLAI”, and “Understory_LAI_biomass”. All directories contain figures of the data and the R code to read and output the table data in the comma-separated values (CSV) format. In the data file, unavailable data are shown in NA.

Tree_census
- /Figures

- TreeCensus.R

- TreeCensus.csv
The items recorded are tree location, species name, DBH in the year 2010, tree height in the year 2010, canopy maximum projections (north, west, south, east) in the year 2010, lowest green foliation height in the year 2010, DBH in the year 2014, tree height in the year 2014.

Tree_allometry
- /Figures

- /Photos
- Tree_allometry.R
- allometry_summary.csv
This file contains the summary of vertically integrated values. The record of 22 trees is summarized.  

- Branch_drymass_profile.csv
Vertical profiles of dry biomass of dead and live branches in every 50 cm interval are recorded.

- Leafarea_profile.csv
Vertical profiles of green leaf area in every 50 cm interval are recorded.

- Leafmass_profile.csv
Vertical profiles of green leaf biomass in every 50 cm interval are recorded.

- stem_drymass_profile.csv
Vertical profiles of stem biomass in every 50 cm interval are recorded.

- stem_diameter_profile.csv
Vertical profiles of stem diameter in the lowest, middle and top of every 50 cm interval are recorded.

OverstoryLAI
- /Figures
- /Photo
The upward and downward photographs at the gap fraction measurement locations are stored. The file names “A1.jpg”, “A1up.jpg”, and “A1dn.jpg” indicate location view, upward and downward images in the sampling location at the first point of A-line. The line name and the location numbers are corresponding to those in the data file “GapFraction_20180724.csv”

- /Geolocation_shape
Geolocation of all gap fraction measurements is recorded as GIS vector data (point data). The file format is an ESRI shape file.

- GapFraction_20180724.R

- GapFraction_20180724.csv
Gap fraction measurements in four transects (A, B, C, and D) are recorded at five view directions (ring1 to ring 5).

- PAI_20180724.csv
Landscape-scale Plant Area Index (PAI) and Leaf Area Index (LAI) as calculated by the method in Section 3.3. 

- SkyCondition_20180724.csv
This is ancillary data used in PAI and LAI estimation by the method described in 3.3.

Understory_LAI_biomass
- /Figures

- /Photos

-Understory_LAI_biomass.R

- Location_info.csv
Geolocation of 16 quadrat sites in latitude/longitude (WGS 84) and UTM zone 6N (Easting/Northing)

- Understory_LAI_biomass.csv
LAI and total biomass are recorded as well as green leaf and woody biomass of CloudBerry, Cotton Grass, Bog BilBerry, CranBerry, Bog Birch, Black Spruce, Labrador Tea, and others. 

- LMA.csv
	LMA of seven dominant species are recorded. 

