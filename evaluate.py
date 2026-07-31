from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score
import pandas as pd

# 1. Initialize the new models (using balanced weights where possible)
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
# Note: GradientBoosting doesn't have a built-in 'class_weight', so we evaluate its raw performance
gbc = GradientBoostingClassifier(random_state=42)

# 2. Train the models
print("Training Logistic Regression...")
log_reg.fit(X_train, y_train)

print("Training Gradient Boosting...")
gbc.fit(X_train, y_train)

# 3. Make Predictions
y_pred_lr = log_reg.predict(X_test)
y_pred_gbc = gbc.predict(X_test)
y_pred_rf = best_rf.predict(X_test) # Your tuned RF from earlier

# 4. Compile a Comparative Results Table
results = {
    'Model': ['Logistic Regression', 'Gradient Boosting', 'Tuned Random Forest'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr), 
        accuracy_score(y_test, y_pred_gbc), 
        accuracy_score(y_test, y_pred_rf)
    ],
    'F1-Score (Macro)': [
        f1_score(y_test, y_pred_lr, average='macro'), 
        f1_score(y_test, y_pred_gbc, average='macro'), 
        f1_score(y_test, y_pred_rf, average='macro')
    ],
    'Recall (In Danger Class)': [
        recall_score(y_test, y_pred_lr, pos_label=1), 
        recall_score(y_test, y_pred_gbc, pos_label=1), 
        recall_score(y_test, y_pred_rf, pos_label=1)
    ]
}

comparison_df = pd.DataFrame(results)
print("\n--- Model Comparison Results ---")
print(comparison_df.to_string(index=False))

# Save this table for your paper
comparison_df.to_csv('/content/UNESCO_ML_Project/outputs/model_comparison.csv', index=False)
