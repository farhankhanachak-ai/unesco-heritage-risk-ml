import pandas as pd
import numpy as np

file_path = '/content/UNESCO_ML_Project/data/unesco_world_heritage_sites.csv'
df = pd.read_csv(file_path)

# 1. Feature Selection: Keep only the columns relevant to predicting Danger Status
# Use 'danger_status' instead of 'danger'
columns_to_keep = [
    'name', 'longitude', 'latitude', 'area_hectares', 
    'category', 'danger_status', 'date_inscribed'
]

existing_columns = [col for col in columns_to_keep if col in df.columns]
if 'name' not in existing_columns and 'name_en' in df.columns:
    existing_columns.append('name_en')
df = df[existing_columns]

# 2. Handle Missing Values
if 'longitude' in df.columns and 'latitude' in df.columns:
    df = df.dropna(subset=['longitude', 'latitude'])

if 'area_hectares' in df.columns:
    df['area_hectares'] = pd.to_numeric(df['area_hectares'], errors='coerce')
    df['area_hectares'] = df['area_hectares'].fillna(df['area_hectares'].median())

# 3. Encoding Categorical Variables
# Ensure we refer to 'danger_status' here as well
if 'danger_status' in df.columns:
    df['danger_status'] = df['danger_status'].astype(int)
    # Rename it to 'danger' for simplicity in later steps
    df = df.rename(columns={'danger_status': 'danger'})

if 'category' in df.columns:
    df = pd.get_dummies(df, columns=['category'], drop_first=False)

# 4. Save the cleaned dataset
clean_path = '/content/UNESCO_ML_Project/data/unesco_cleaned.csv'
df.to_csv(clean_path, index=False)

print("✅ Data successfully cleaned and saved to:", clean_path)
display(df.head())
