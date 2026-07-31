# unesco-heritage-risk-ml
A comparative machine learning analysis of geospatial vulnerability in UNESCO World Heritage Sites.
# Geospatial Vulnerability in UNESCO World Heritage Sites

This repository contains the dataset, machine learning models, and academic manuscript for predicting the endangerment status of UNESCO World Heritage Sites.

## Project Overview
A comparative analysis addressing the extreme class imbalance (~4% minority class) inherent in heritage conservation data. The project evaluates Logistic Regression, Gradient Boosting, and Cost-Sensitive Random Forest architectures to overcome the "Accuracy Paradox."

## Core Findings
* **Optimal Model:** Tuned Random Forest utilizing algorithm-level class weighting.
* **Performance:** 86.0% Accuracy, 0.55 Macro F1-Score.
* **Feature Importance:** Absolute geographical coordinates (Latitude and Longitude) account for 69% of the predictive variance, mathematically highlighting regional geopolitical instability.

## Repository Setup
To replicate the environment and run the model training scripts:

1. Clone the repository:
   `git clone https://github.com/yourusername/unesco-heritage-risk-ml.git`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run the Colab notebook located in `/notebooks/` or execute the Python scripts in `/src/`.

## Authors
* **Farhan Khan** - *Department of Computer Engineering, BUITEMS*
* **Prof. Sibghat Ullah Bazai** (Advisor)
