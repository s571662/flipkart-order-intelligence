from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)



from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "orders_dataset.csv"


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------

X = df.drop(columns=["returned", "order_id"])
y = df["returned"]


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)


print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))

print("\nTraining return rate:")
print(round(y_train.mean(), 4))

print("\nTest return rate:")
print(round(y_test.mean(), 4))


# ---------------------------------------------------------
# Define feature types
# ---------------------------------------------------------

numeric_features = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
]

categorical_features = [
    "product_category",
    "payment_method",
]


# ---------------------------------------------------------
# Numeric preprocessing
# ---------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


# ---------------------------------------------------------
# Categorical preprocessing
# ---------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)


# ---------------------------------------------------------
# Combined preprocessing
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# ---------------------------------------------------------
# Fit preprocessing ONLY on training data
# ---------------------------------------------------------

X_train_transformed = preprocessor.fit_transform(X_train)

X_test_transformed = preprocessor.transform(X_test)


print("\nPreprocessing complete.")

print(
    "Training transformed shape:",
    X_train_transformed.shape,
)

print(
    "Test transformed shape:",
    X_test_transformed.shape,
)

print(
    "Test transformed shape:",
    X_test_transformed.shape,
)
# ---------------------------------------------------------
# Dummy Classifier Baseline
# ---------------------------------------------------------

dummy = DummyClassifier(
    strategy="most_frequent"
)

dummy.fit(X_train_transformed, y_train)

dummy_pred = dummy.predict(X_test_transformed)

dummy_accuracy = accuracy_score(
    y_test,
    dummy_pred
)

dummy_f1 = f1_score(
    y_test,
    dummy_pred,
    zero_division=0
)

print("\n" + "=" * 60)
print("DUMMY CLASSIFIER RESULTS")
print("=" * 60)

print(f"Accuracy: {dummy_accuracy:.4f}")
print(f"F1 Score (returned=1): {dummy_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        dummy_pred,
        zero_division=0
    )
)

# ---------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------

logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_transformed,
    y_train
)

# Default threshold = 0.5

logistic_probabilities = logistic_model.predict_proba(
    X_test_transformed
)[:, 1]

logistic_predictions = (
    logistic_probabilities >= 0.5
).astype(int)


# ---------------------------------------------------------
# Logistic Regression Metrics
# ---------------------------------------------------------

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

logistic_f1 = f1_score(
    y_test,
    logistic_predictions
)

logistic_recall = recall_score(
    y_test,
    logistic_predictions
)

logistic_precision = precision_score(
    y_test,
    logistic_predictions
)

logistic_roc_auc = roc_auc_score(
    y_test,
    logistic_probabilities
)


print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS - THRESHOLD 0.5")
print("=" * 60)

print(f"Accuracy:  {logistic_accuracy:.4f}")
print(f"F1:        {logistic_f1:.4f}")
print(f"Recall:    {logistic_recall:.4f}")
print(f"Precision: {logistic_precision:.4f}")
print(f"ROC-AUC:   {logistic_roc_auc:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        logistic_predictions,
        zero_division=0
    )
)

# ---------------------------------------------------------
# Logistic Regression Threshold Sweep
# ---------------------------------------------------------

threshold_results = []

for threshold in np.arange(0.10, 0.91, 0.01):
    threshold = round(threshold, 2)
    
    threshold_predictions = (
        logistic_probabilities >= threshold
    ).astype(int)

    threshold_f1 = f1_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    threshold_precision = precision_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "f1": threshold_f1,
        "recall": threshold_recall,
        "precision": threshold_precision,
    })


threshold_df = pd.DataFrame(
    threshold_results
)

best_threshold_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = best_threshold_row["threshold"]
best_threshold_f1 = best_threshold_row["f1"]
best_threshold_recall = best_threshold_row["recall"]
best_threshold_precision = best_threshold_row["precision"]


print("\n" + "=" * 60)
print("LOGISTIC REGRESSION THRESHOLD SWEEP")
print("=" * 60)

print(
    threshold_df.to_string(
        index=False,
        formatters={
            "threshold": "{:.2f}".format,
            "f1": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "precision": "{:.4f}".format,
        }
    )
)

print("\nBest F1 threshold:")
print(f"Threshold: {best_threshold:.2f}")
print(f"F1:        {best_threshold_f1:.4f}")
print(f"Recall:    {best_threshold_recall:.4f}")
print(f"Precision: {best_threshold_precision:.4f}")

# ---------------------------------------------------------
# Random Forest Pipeline
# ---------------------------------------------------------

random_forest = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
)

rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            random_forest,
        ),
    ]
)

# ---------------------------------------------------------
# Random Forest Grid Search
# ---------------------------------------------------------

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [6, 10, None],
}

# ---------------------------------------------------------
# Random Forest - 5 Fold Cross Validation
# ---------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# ---------------------------------------------------------
# Random Forest Grid Search
# ---------------------------------------------------------

random_forest = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", random_forest)
    ]
)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [6, 10, None],
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(
    X_train,
    y_train
)

print("\n" + "=" * 60)
print("RANDOM FOREST GRID SEARCH RESULTS")
print("=" * 60)

print("Best parameters:")
print(grid_search.best_params_)

print(
    f"Best cross-validated ROC-AUC: "
    f"{grid_search.best_score_:.4f}"
)

# ---------------------------------------------------------
# Evaluate Winning Random Forest on Test Set
# ---------------------------------------------------------

best_rf_pipeline = grid_search.best_estimator_

rf_test_probabilities = (
    best_rf_pipeline.predict_proba(X_test)[:, 1]
)

rf_test_predictions = (
    rf_test_probabilities >= 0.5
).astype(int)

rf_test_accuracy = accuracy_score(
    y_test,
    rf_test_predictions
)

rf_test_f1 = f1_score(
    y_test,
    rf_test_predictions,
    zero_division=0
)

rf_test_recall = recall_score(
    y_test,
    rf_test_predictions,
    zero_division=0
)

rf_test_precision = precision_score(
    y_test,
    rf_test_predictions,
    zero_division=0
)

rf_test_roc_auc = roc_auc_score(
    y_test,
    rf_test_probabilities
)

print("\n" + "=" * 60)
print("WINNING RANDOM FOREST - TEST RESULTS")
print("=" * 60)

print(f"Accuracy:  {rf_test_accuracy:.4f}")
print(f"F1:        {rf_test_f1:.4f}")
print(f"Recall:    {rf_test_recall:.4f}")
print(f"Precision: {rf_test_precision:.4f}")
print(f"ROC-AUC:   {rf_test_roc_auc:.4f}")

print(
    f"\nCV ROC-AUC:   {grid_search.best_score_:.4f}"
)

print(
    f"Test ROC-AUC: {rf_test_roc_auc:.4f}"
)

print(
    f"Difference:   "
    f"{abs(grid_search.best_score_ - rf_test_roc_auc):.4f}"
)

# ---------------------------------------------------------
# Random Forest Feature Importance
# ---------------------------------------------------------

best_preprocessor = (
    best_rf_pipeline.named_steps["preprocessor"]
)

best_rf_model = (
    best_rf_pipeline.named_steps["model"]
)

feature_names = (
    best_preprocessor.get_feature_names_out()
)

feature_importances = (
    best_rf_model.feature_importances_
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importances
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance_df.head(10).to_string(
        index=False
    )
)

# ---------------------------------------------------------
# Permutation Importance
# ---------------------------------------------------------

permutation_result = permutation_importance(
    best_rf_pipeline,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

permutation_df = pd.DataFrame({
    "feature": X_test.columns,
    "permutation_importance": permutation_result.importances_mean
})

permutation_df = permutation_df.sort_values(
    "permutation_importance",
    ascending=False
)

print("\n" + "=" * 60)
print("PERMUTATION IMPORTANCE")
print("=" * 60)

print(
    permutation_df.head(10).to_string(
        index=False
    )
)

# ---------------------------------------------------------
# Compare Impurity vs Permutation Importance
# ---------------------------------------------------------

top5_impurity = importance_df.head(5).copy()

top5_impurity["original_feature"] = (
    top5_impurity["feature"]
    .str.replace("numeric__", "", regex=False)
    .str.replace("categorical__", "", regex=False)
)

comparison = top5_impurity[
    ["feature", "importance"]
].rename(
    columns={
        "feature": "impurity_feature",
        "importance": "impurity_importance"
    }
)

comparison["permutation_importance"] = (
    comparison["impurity_feature"]
    .str.replace("numeric__", "", regex=False)
    .str.replace("categorical__", "", regex=False)
    .map(
        permutation_df.set_index("feature")[
            "permutation_importance"
        ]
    )
)

print("\n" + "=" * 60)
print("TOP-5 IMPURITY VS PERMUTATION IMPORTANCE")
print("=" * 60)

print(
    comparison.to_string(index=False)
)

# ---------------------------------------------------------
# Subgroup Analysis
# ---------------------------------------------------------

def calculate_subgroup_metrics(
    df,
    predictions,
    actual,
    column
):
    results = []

    for group in sorted(df[column].dropna().unique()):

        mask = df[column] == group

        group_actual = actual[mask]
        group_predictions = predictions[mask]

        group_recall = recall_score(
            group_actual,
            group_predictions,
            zero_division=0
        )

        group_precision = precision_score(
            group_actual,
            group_predictions,
            zero_division=0
        )

        results.append({
            column: group,
            "count": mask.sum(),
            "recall": group_recall,
            "precision": group_precision
        })

    return pd.DataFrame(results)


product_results = calculate_subgroup_metrics(
    X_test,
    rf_test_predictions,
    y_test,
    "product_category"
)

payment_results = calculate_subgroup_metrics(
    X_test,
    rf_test_predictions,
    y_test,
    "payment_method"
)

print("\n" + "=" * 60)
print("SUBGROUP PERFORMANCE - PRODUCT CATEGORY")
print("=" * 60)

print(
    product_results.to_string(index=False)
)

print("\n" + "=" * 60)
print("SUBGROUP PERFORMANCE - PAYMENT METHOD")
print("=" * 60)

print(
    payment_results.to_string(index=False)
)

# ---------------------------------------------------------
# Random Forest Threshold Sweep
# ---------------------------------------------------------

rf_threshold_results = []

for threshold in np.arange(0.10, 0.91, 0.01):

    predictions = (
        rf_test_probabilities >= threshold
    ).astype(int)

    threshold_f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    threshold_precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    rf_threshold_results.append({
        "threshold": threshold,
        "f1": threshold_f1,
        "recall": threshold_recall,
        "precision": threshold_precision
    })


rf_threshold_df = pd.DataFrame(
    rf_threshold_results
)

best_rf_threshold_row = (
    rf_threshold_df
    .sort_values("f1", ascending=False)
    .iloc[0]
)

t_rf = float(
    best_rf_threshold_row["threshold"]
)

print("\n" + "=" * 60)
print("RANDOM FOREST THRESHOLD SWEEP")
print("=" * 60)

print(f"t*_rf:      {t_rf:.2f}")
print(
    f"Best F1:    "
    f"{best_rf_threshold_row['f1']:.4f}"
)
print(
    f"Recall:     "
    f"{best_rf_threshold_row['recall']:.4f}"
)
print(
    f"Precision:  "
    f"{best_rf_threshold_row['precision']:.4f}"
)

# ---------------------------------------------------------
# Save Final Return Risk Model
# ---------------------------------------------------------

joblib.dump(
    best_rf_pipeline,
    "models/return_risk_model.pkl"
)

print("\nSaved final model to:")
print("models/return_risk_model.pkl")

# ---------------------------------------------------------
# Save Random Forest Threshold
# ---------------------------------------------------------

with open(
    "models/return_risk_threshold.txt",
    "w"
) as f:
    f.write(str(t_rf))

print(
    "Saved threshold to "
    "models/return_risk_threshold.txt"
)

# ---------------------------------------------------------
# Verify Saved Model
# ---------------------------------------------------------

loaded_model = joblib.load(
    "models/return_risk_model.pkl"
)

loaded_probabilities = (
    loaded_model.predict_proba(X_test)[:, 1]
)

print("\n" + "=" * 60)
print("SAVED MODEL VERIFICATION")
print("=" * 60)

print("Model loaded successfully.")
print(
    "Pipeline type:",
    type(loaded_model).__name__
)

print(
    "First 5 probabilities:",
    loaded_probabilities[:5]
)