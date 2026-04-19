# Data Sources

This file documents all datasets, imagery products, labels, and derived outputs used in the **Hula Valley Land Cover Classification** project.

---

# 1. Study Area Boundary

## Area of Interest (AOI)

The study area represents the **Hula Valley**, located in northern Israel.

Used in Google Earth Engine as a polygon / feature collection for:

- image clipping
- sampling region
- export boundary
- map centering

---

# 2. Satellite Imagery

## Sentinel-2 Surface Reflectance Harmonized

**Dataset ID (GEE):**

COPERNICUS/S2_SR_HARMONIZED

Used as the primary multispectral imagery source.

### Date Range Used

- Start: 2024-04-01
- End: 2024-10-31

### Spatial Resolution

10 m (selected bands)

### Bands Used

- B2 (Blue)
- B3 (Green)
- B4 (Red)
- B8 (NIR)
- B11 (SWIR1)
- B12 (SWIR2)

### Preprocessing Applied

- Cloud filtering (<30% cloudy pixels)
- Scene Classification Layer masking
- Reflectance scaling (/10000)
- Median seasonal compositing

:contentReference[oaicite:0]{index=0}

---

# 3. Derived Spectral Indices

The following indices were generated from Sentinel-2 bands:

## NDVI

Normalized Difference Vegetation Index

Used to enhance vegetation detection.

## NDWI

Normalized Difference Water Index

Used to improve water body identification.

## MNDWI

Modified Normalized Difference Water Index

Used for open water and wetland extraction.

## NDBI

Normalized Difference Built-up Index

Used to detect built surfaces and infrastructure.

:contentReference[oaicite:1]{index=1}

---

# 4. Land Cover Labels

## Dynamic World v1

**Dataset ID (GEE):**

GOOGLE/DYNAMICWORLD/V1

Used as training labels for supervised classification.

### Method Used

- All Dynamic World label images within the date range were collected.
- Pixel-wise modal class was calculated using `.mode()`.

### Classes Used

0. Water  
1. Trees  
2. Grass  
3. Flooded Vegetation  
4. Crops  
5. Shrub & Scrub  
6. Built Area  
7. Bare Ground  
8. Snow / Ice

:contentReference[oaicite:2]{index=2}

---

# 5. Machine Learning Inputs

## Predictor Variables

Final feature stack used for training:

- B2
- B3
- B4
- B8
- B11
- B12
- NDVI
- NDWI
- MNDWI
- NDBI

## Sample Extraction

Stratified random sampling used across classes:

- 1500 total points
- 70% training
- 30% testing

:contentReference[oaicite:3]{index=3}

---

# 6. Classification Models

## Random Forest

Parameters:

- 200 trees
- bag fraction = 0.7
- seed = 42

## Support Vector Machine (SVM)

Parameters:

- Kernel: RBF
- gamma = 0.5
- cost = 10

:contentReference[oaicite:4]{index=4}

---

# 7. Exported Outputs

Generated outputs include:

## Tables

- Hula_S2_DynamicWorld_samples.csv

## Raster Outputs

- Hula_RF_classification
- Hula_SVM_classification
- Hula_DW_label_mode
- Hula_DW_label_mode_clean
- Hula_DW_label_visualized

:contentReference[oaicite:5]{index=5}

---

# 8. Software Environment

## Platforms Used

- Google Earth Engine
- Python
- GitHub

## Typical Python Libraries

- pandas
- numpy
- scikit-learn
- matplotlib
- rasterio (optional)

---

# 9. Notes

- Dynamic World labels are probabilistic machine-generated products and may contain local inaccuracies.
- Seasonal composites reduce cloud noise but may smooth short-term land cover variation.
- Snow/Ice class likely has minimal relevance in the Hula Valley and may be rare or absent.

---

# 10. Citation Suggestions

ESA Copernicus Sentinel-2 Mission  
Google Dynamic World Dataset  
Google Earth Engine Platform

---