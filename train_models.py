import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset
clean_path = '/content/UNESCO_ML_Project/data/unesco_cleaned.csv'
df = pd.read_csv(clean_path)

# Prepare Features (X) and Target (y)
# We drop non-predictive text columns and the target column itself
X = df.drop(columns=['danger', 'name', 'name_en', 'date_inscribed'], errors='ignore')
y = df['danger']

# Train-Test Split (80% training, 20% testing)
# stratify=y is crucial here: it ensures the tiny percentage of 'In Danger' sites 
# is distributed evenly between the training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} sites")
print(f"Testing set: {X_test.shape[0]} sites\n")

# Initialize and Train the Model
print("Training Random Forest Classifier (with balanced class weights)...")
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Make Predictions on the test set
y_pred = rf_model.predict(X_test)

# --- Evaluation Metrics ---
print("\n--- Classification Report ---")
# The report focuses on Precision, Recall, and F1-Score (much better than raw accuracy)
print(classification_report(y_test, y_pred))

# Generate and save the Confusion Matrix visual
print("\nGenerating Figure 4: Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Safe (0)', 'In Danger (1)'], 
            yticklabels=['Safe (0)', 'In Danger (1)'])
plt.title('Random Forest Confusion Matrix', fontsize=14)
plt.ylabel('Actual Status', fontsize=12)
plt.xlabel('Predicted Status', fontsize=12)
plt.tight_layout()

# Save for the paper/presentation
plt.savefig('/content/UNESCO_ML_Project/outputs/figures/fig4_confusion_matrix.png', dpi=300)
plt.show()
