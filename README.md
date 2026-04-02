### XGBoost vs. MLP: Forest Biomass Estimation in South Sulawesi
This repository contains a comparative study between eXtreme Gradient Boosting (XGBoost) and Multi-Layer Perceptron (MLP) for estimating Aboveground Biomass (AGB) in South Sulawesi, Indonesia. The project leverages satellite-derived data and GEDI products processed via Google Earth Engine (GEE).

Estimating forest biomass is crucial for carbon accounting and climate change mitigation. This project evaluates the performance of a tree-based ensemble model (XGBoost) against a deep learning approach (MLP) using high-dimensional satellite embeddings and GEDI biomass data.

📊 Data Sources
All datasets were retrieved and pre-processed using Google Earth Engine (GEE). To maintain consistency, all satellite layers were resampled to a spatial resolution of 1000 meters, matching the GEDI L4B grid.

- Satellite Embedding V1 (Feature extraction for model input)
  https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- GEDI L4B Gridded AGB Density (Version 2) (Target variable for training/validation)
  https://developers.google.com/earth-engine/datasets/catalog/LARSE_GEDI_GEDI04_B_002
- Dynamic World V1 (Masking to isolate forest-only pixels)
  https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1

<p align="center">
  <img src="https://github.com/diniyrti/XGBoost-vs-MLP-Forest-Biomass-Estimation/blob/main/images/GEDI%20L4B%20Dataset.png?raw=true" width="500">
</p>

⚙️ Methodology
Pre-processing Steps:
1. Masking: Used Dynamic World V1 to filter only forest-classified pixels.
2. Alignment: Resampled all predictor layers to 1000m resolution to align with GEDI L4B.
3. Normalization: Standardized features for optimal MLP performance.

Two machine learning architectures were compared:
- XGBoost: A gradient-boosted decision tree framework known for its efficiency on tabular data.
- MLP (Multi-Layer Perceptron): A feed-forward artificial neural network designed to capture non-linear relationships.

📈 Results
Based on the experimental results, the MLP model outperformed XGBoost, showing higher accuracy and lower error rates in predicting forest biomass density. The MLP model achieved a Root Mean Square Error (RMSE) of 79.935, which is lower than the 80.396 recorded by XGBoost. This indicates that the neural network approach was more effective at minimizing the deviations between the predicted values and the GEDI L4B ground truth.In terms of the correlation between predicted and observed values, the MLP also yielded a higher Coefficient of Determination ($R^2$) of 0.643, compared to XGBoost's 0.639. This suggests that approximately 64.3% of the biomass variance in the study area can be explained by the MLP model using the provided satellite embeddings.

![image alt](https://github.com/diniyrti/XGBoost-vs-MLP-Forest-Biomass-Estimation/blob/main/images/Performance%20Model.png)

![image alt](https://github.com/diniyrti/XGBoost-vs-MLP-Forest-Biomass-Estimation/blob/main/images/Forest%20Biomass%20Prediction.png)

The MLP model demonstrated a better ability to generalize the complex relationship between satellite embeddings and biomass density in the tropical landscape of South Sulawesi
