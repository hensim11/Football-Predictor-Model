# Premier League Match Outcome Prediction Model
# ------------------------------------------------------------
# This script:
# 1. Loads multiple Premier League CSV files
# 2. Cleans and combines them
# 3. Builds pre-match features using only past information
# 4. Trains TWO weighted models:
#       - Weighted Logistic Regression
#       - Weighted Random Forest
# 5. Handles class imbalance using class weights
# 6. Evaluates both on the latest season
# 7. Compares the models using multiple metrics
# 8. Displays confusion matrices for:
#       - Weighted Logistic Regression
#       - Weighted Random Forest
# 9. Predicts outcomes for new fixtures using the better model
#
# Libraries needed:
# pip install pandas numpy scikit-learn matplotlib
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import glob
import os
from collections import defaultdict, deque

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_class_weight


# ------------------------------------------------------------
# 1) LOAD ALL PREMIER LEAGUE CSV FILES
# ------------------------------------------------------------
# Put this script in the same folder as the CSV files,
# or change the pattern below to match your file location.
DATA_PATTERN = "E0*.csv"

files = sorted(glob.glob(DATA_PATTERN))

if not files:
    raise FileNotFoundError(
        "No CSV files found. Put the CSV files in the same folder as this script "
        "or change DATA_PATTERN to the correct path."
    )

all_dfs = []

for file in files:
    df = pd.read_csv(file)
    df["SourceFile"] = os.path.basename(file)
    all_dfs.append(df)

matches = pd.concat(all_dfs, ignore_index=True, sort=False)


# ------------------------------------------------------------
# 2) BASIC CLEANING
# ------------------------------------------------------------
required_columns = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "SourceFile"
]

# Create missing required columns if any are absent
for col in required_columns:
    if col not in matches.columns:
        matches[col] = np.nan

matches = matches[required_columns].copy()

# Convert dates
matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")

# Remove rows missing essential fields
matches = matches.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])

# Keep only valid target classes
matches = matches[matches["FTR"].isin(["H", "D", "A"])].copy()

# Convert numeric columns safely
numeric_cols = ["FTHG", "FTAG", "HS", "AS", "HST", "AST"]
for col in numeric_cols:
    matches[col] = pd.to_numeric(matches[col], errors="coerce").fillna(0)

# Sort in chronological order
matches = matches.sort_values("Date").reset_index(drop=True)


# ------------------------------------------------------------
# 3) HELPER FUNCTIONS
# ------------------------------------------------------------
def get_points(result, side):
    """
    Returns the points won by a team based on match result.
    result: 'H', 'D', 'A'
    side: 'home' or 'away'
    """
    if result == "D":
        return 1
    if result == "H" and side == "home":
        return 3
    if result == "A" and side == "away":
        return 3
    return 0


def safe_avg(total, count, default):
    """
    Returns total / count if count > 0, otherwise returns a default.
    """
    return total / count if count > 0 else default


# ------------------------------------------------------------
# 4) FEATURE ENGINEERING
# ------------------------------------------------------------
def build_pre_match_features(df):
    """
    Builds pre-match features using only information available
    before each match is played. This avoids data leakage.
    """

    # Team stats across all previous matches
    overall_stats = defaultdict(lambda: {
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0,
        "gf": 0,
        "ga": 0,
        "shots": 0,
        "sot": 0
    })

    # Team stats within the current season
    season_stats = defaultdict(lambda: defaultdict(lambda: {
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0,
        "gf": 0,
        "ga": 0,
        "shots": 0,
        "sot": 0
    }))

    # Recent form trackers
    recent_points = defaultdict(lambda: deque(maxlen=5))
    recent_goal_diff = defaultdict(lambda: deque(maxlen=5))

    feature_rows = []

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        season = row["SourceFile"]

        h_all = overall_stats[home]
        a_all = overall_stats[away]
        h_season = season_stats[season][home]
        a_season = season_stats[season][away]

        # Build pre-match features
        feature_row = {
            "Date": row["Date"],
            "Season": season,
            "HomeTeam": home,
            "AwayTeam": away,
            "FTR": row["FTR"],

            # Overall historical features
            "home_ppg": safe_avg(h_all["points"], h_all["played"], 1.30),
            "away_ppg": safe_avg(a_all["points"], a_all["played"], 1.30),

            "home_win_rate": safe_avg(h_all["wins"], h_all["played"], 0.33),
            "away_win_rate": safe_avg(a_all["wins"], a_all["played"], 0.33),

            "home_draw_rate": safe_avg(h_all["draws"], h_all["played"], 0.25),
            "away_draw_rate": safe_avg(a_all["draws"], a_all["played"], 0.25),

            "home_avg_gf": safe_avg(h_all["gf"], h_all["played"], 1.30),
            "away_avg_gf": safe_avg(a_all["gf"], a_all["played"], 1.30),

            "home_avg_ga": safe_avg(h_all["ga"], h_all["played"], 1.30),
            "away_avg_ga": safe_avg(a_all["ga"], a_all["played"], 1.30),

            "home_avg_shots": safe_avg(h_all["shots"], h_all["played"], 12.0),
            "away_avg_shots": safe_avg(a_all["shots"], a_all["played"], 12.0),

            "home_avg_sot": safe_avg(h_all["sot"], h_all["played"], 4.0),
            "away_avg_sot": safe_avg(a_all["sot"], a_all["played"], 4.0),

            # Current season features
            "home_season_ppg": safe_avg(h_season["points"], h_season["played"], 1.30),
            "away_season_ppg": safe_avg(a_season["points"], a_season["played"], 1.30),

            "home_season_avg_gf": safe_avg(h_season["gf"], h_season["played"], 1.30),
            "away_season_avg_gf": safe_avg(a_season["gf"], a_season["played"], 1.30),

            "home_season_avg_ga": safe_avg(h_season["ga"], h_season["played"], 1.30),
            "away_season_avg_ga": safe_avg(a_season["ga"], a_season["played"], 1.30),

            # Last 5 matches form
            "home_form_pts5": np.mean(recent_points[home]) if recent_points[home] else 1.20,
            "away_form_pts5": np.mean(recent_points[away]) if recent_points[away] else 1.20,

            "home_form_gd5": np.mean(recent_goal_diff[home]) if recent_goal_diff[home] else 0.0,
            "away_form_gd5": np.mean(recent_goal_diff[away]) if recent_goal_diff[away] else 0.0,
        }

        # Difference features
        diff_pairs = [
            ("ppg_diff", "home_ppg", "away_ppg"),
            ("win_rate_diff", "home_win_rate", "away_win_rate"),
            ("draw_rate_diff", "home_draw_rate", "away_draw_rate"),
            ("avg_gf_diff", "home_avg_gf", "away_avg_gf"),
            ("avg_ga_diff", "home_avg_ga", "away_avg_ga"),
            ("avg_shots_diff", "home_avg_shots", "away_avg_shots"),
            ("avg_sot_diff", "home_avg_sot", "away_avg_sot"),
            ("season_ppg_diff", "home_season_ppg", "away_season_ppg"),
            ("season_avg_gf_diff", "home_season_avg_gf", "away_season_avg_gf"),
            ("season_avg_ga_diff", "home_season_avg_ga", "away_season_avg_ga"),
            ("form_pts5_diff", "home_form_pts5", "away_form_pts5"),
            ("form_gd5_diff", "home_form_gd5", "away_form_gd5"),
        ]

        for diff_name, left, right in diff_pairs:
            feature_row[diff_name] = feature_row[left] - feature_row[right]

        feature_rows.append(feature_row)

        # Update team stats AFTER storing the pre-match features
        ftr = row["FTR"]
        fthg = row["FTHG"]
        ftag = row["FTAG"]
        hs = row["HS"]
        a_s = row["AS"]
        hst = row["HST"]
        ast = row["AST"]

        for team, gf, ga, shots, sot, side in [
            (home, fthg, ftag, hs, hst, "home"),
            (away, ftag, fthg, a_s, ast, "away")
        ]:
            pts = get_points(ftr, side)

            for stat_dict in (overall_stats[team], season_stats[season][team]):
                stat_dict["played"] += 1
                stat_dict["gf"] += gf
                stat_dict["ga"] += ga
                stat_dict["shots"] += shots
                stat_dict["sot"] += sot
                stat_dict["points"] += pts

                if pts == 3:
                    stat_dict["wins"] += 1
                elif pts == 1:
                    stat_dict["draws"] += 1
                else:
                    stat_dict["losses"] += 1

        recent_points[home].append(get_points(ftr, "home"))
        recent_points[away].append(get_points(ftr, "away"))

        recent_goal_diff[home].append(fthg - ftag)
        recent_goal_diff[away].append(ftag - fthg)

    return pd.DataFrame(feature_rows), overall_stats, season_stats, recent_points, recent_goal_diff


feature_df, overall_stats, season_stats, recent_points, recent_goal_diff = build_pre_match_features(matches)


# ------------------------------------------------------------
# 5) TRAIN / TEST SPLIT
# ------------------------------------------------------------
# Use the latest season as the test set
latest_season = feature_df["Season"].iloc[-1]

train_df = feature_df[feature_df["Season"] != latest_season].copy()
test_df = feature_df[feature_df["Season"] == latest_season].copy()

X_train = train_df.drop(columns=["FTR", "Date", "Season"])
y_train = train_df["FTR"]

X_test = test_df.drop(columns=["FTR", "Date", "Season"])
y_test = test_df["FTR"]


# ------------------------------------------------------------
# 6) EXAMINE CLASS IMBALANCE
# ------------------------------------------------------------
print("=" * 80)
print("CLASS DISTRIBUTION IN TRAINING DATA")
print("=" * 80)
print(y_train.value_counts().sort_index())
print()

# Compute balanced class weights from the training labels only
classes = np.array(sorted(y_train.unique()))
weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, weights_array))

print("Computed class weights:")
for cls, weight in class_weights.items():
    print(f"  {cls}: {weight:.4f}")
print("=" * 80)


# ------------------------------------------------------------
# 7) PREPROCESSING
# ------------------------------------------------------------
# Keep preprocessing the same for both models so comparison is fair.
numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = ["HomeTeam", "AwayTeam"]

# Logistic regression benefits from scaling
preprocessor_for_logistic = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        ),
    ]
)

# Random forest does not need scaling, but we still need imputation/encoding
preprocessor_for_rf = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numeric_features
        ),
        (
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        ),
    ]
)


# ------------------------------------------------------------
# 8) BUILD WEIGHTED MODELS
# ------------------------------------------------------------
weighted_logistic_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_for_logistic),
    ("model", LogisticRegression(
        max_iter=5000,
        class_weight=class_weights,
        random_state=42
    ))
])

weighted_random_forest_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_for_rf),
    ("model", RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    ))
])


# ------------------------------------------------------------
# 9) TRAIN MODELS
# ------------------------------------------------------------
print("TRAINING WEIGHTED LOGISTIC REGRESSION...")
weighted_logistic_pipeline.fit(X_train, y_train)

print("TRAINING WEIGHTED RANDOM FOREST...")
weighted_random_forest_pipeline.fit(X_train, y_train)

print("TRAINING COMPLETE.")
print("=" * 80)


# ------------------------------------------------------------
# 10) EVALUATION FUNCTION
# ------------------------------------------------------------
def evaluate_model(model_name, pipeline, X_test, y_test):
    """
    Evaluates a model and returns useful outputs for comparison.
    """
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    model_classes = pipeline.named_steps["model"].classes_

    accuracy = accuracy_score(y_test, predictions)
    balanced_acc = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")
    multiclass_logloss = log_loss(y_test, probabilities, labels=model_classes)

    # Per-class precision/recall/f1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=["H", "D", "A"],
        zero_division=0
    )

    class_breakdown = pd.DataFrame({
        "Class": ["H", "D", "A"],
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Support": support
    })

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_acc,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
        "Log_Loss": multiclass_logloss
    }

    cm = confusion_matrix(y_test, predictions, labels=["H", "D", "A"])

    print("\n" + "=" * 80)
    print(f"{model_name.upper()} EVALUATION")
    print("=" * 80)
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Balanced Accuracy:  {balanced_acc:.4f}")
    print(f"Macro F1 Score:     {macro_f1:.4f}")
    print(f"Weighted F1 Score:  {weighted_f1:.4f}")
    print(f"Log Loss:           {multiclass_logloss:.4f}")
    print("-" * 80)

    print("Classification report:")
    print(classification_report(y_test, predictions, digits=4, zero_division=0))

    print("Per-class breakdown:")
    print(class_breakdown.to_string(index=False))
    print("-" * 80)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_H", "Actual_D", "Actual_A"],
        columns=["Pred_H", "Pred_D", "Pred_A"]
    )

    print("Confusion Matrix:")
    print(cm_df)
    print("-" * 80)

    return metrics, predictions, probabilities, cm, class_breakdown


# ------------------------------------------------------------
# 11) EVALUATE BOTH WEIGHTED MODELS
# ------------------------------------------------------------
log_metrics, log_preds, log_probs, log_cm, log_class_breakdown = evaluate_model(
    "Weighted Logistic Regression",
    weighted_logistic_pipeline,
    X_test,
    y_test
)

rf_metrics, rf_preds, rf_probs, rf_cm, rf_class_breakdown = evaluate_model(
    "Weighted Random Forest",
    weighted_random_forest_pipeline,
    X_test,
    y_test
)


# ------------------------------------------------------------
# 12) COMPARE MODEL PERFORMANCE
# ------------------------------------------------------------
comparison_df = pd.DataFrame([log_metrics, rf_metrics])

print("\n" + "=" * 80)
print("WEIGHTED MODEL COMPARISON")
print("=" * 80)
print(comparison_df.to_string(index=False))
print("=" * 80)

# Choose best model by balanced accuracy first, then macro F1, then accuracy
best_model_row = comparison_df.sort_values(
    by=["Balanced_Accuracy", "Macro_F1", "Accuracy"],
    ascending=False
).iloc[0]

best_model_name = best_model_row["Model"]

print(f"Best weighted model by this comparison: {best_model_name}")
print("=" * 80)


# ------------------------------------------------------------
# 13) ACTUAL VS PREDICTED COUNTS
# ------------------------------------------------------------
def print_actual_vs_predicted_counts(model_name, y_true, y_pred):
    """
    Prints how many matches of each class actually occurred
    vs how many were predicted.
    """
    actual_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()

    count_summary = pd.DataFrame({
        "Actual_Count": actual_counts,
        "Predicted_Count": pred_counts
    }).fillna(0).astype(int)

    print(f"\nActual vs Predicted Counts - {model_name}")
    print(count_summary)
    print("-" * 80)


print_actual_vs_predicted_counts("Weighted Logistic Regression", y_test, log_preds)
print_actual_vs_predicted_counts("Weighted Random Forest", y_test, rf_preds)


# ------------------------------------------------------------
# 14) BUILD RESULTS TABLES
# ------------------------------------------------------------
def build_prediction_output(test_df, predictions, probabilities, model_classes, model_name):
    """
    Builds a readable prediction output table for each model.
    """
    proba_df = pd.DataFrame(probabilities, columns=model_classes)

    output = test_df[["Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
    output["Predicted"] = predictions
    output["Correct"] = output["FTR"] == output["Predicted"]
    output["Model"] = model_name

    output["Prob_H"] = proba_df["H"] if "H" in proba_df.columns else 0.0
    output["Prob_D"] = proba_df["D"] if "D" in proba_df.columns else 0.0
    output["Prob_A"] = proba_df["A"] if "A" in proba_df.columns else 0.0
    output["Confidence"] = output[["Prob_H", "Prob_D", "Prob_A"]].max(axis=1)

    return output


log_classes = weighted_logistic_pipeline.named_steps["model"].classes_
rf_classes = weighted_random_forest_pipeline.named_steps["model"].classes_

log_results = build_prediction_output(
    test_df,
    log_preds,
    log_probs,
    log_classes,
    "Weighted Logistic Regression"
)

rf_results = build_prediction_output(
    test_df,
    rf_preds,
    rf_probs,
    rf_classes,
    "Weighted Random Forest"
)

print("\nSample predictions - Weighted Logistic Regression:")
print(
    log_results[
        ["Date", "HomeTeam", "AwayTeam", "FTR", "Predicted", "Correct", "Prob_H", "Prob_D", "Prob_A", "Confidence"]
    ].head(10).to_string(index=False)
)

print("\nSample predictions - Weighted Random Forest:")
print(
    rf_results[
        ["Date", "HomeTeam", "AwayTeam", "FTR", "Predicted", "Correct", "Prob_H", "Prob_D", "Prob_A", "Confidence"]
    ].head(10).to_string(index=False)
)


# ------------------------------------------------------------
# 15) PLOT CONFUSION MATRICES
# ------------------------------------------------------------
# These are exactly the two confusion matrices requested:
# - Weighted Logistic Regression
# - Weighted Random Forest
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

log_disp = ConfusionMatrixDisplay(
    confusion_matrix=log_cm,
    display_labels=["H", "D", "A"]
)
log_disp.plot(ax=axes[0], cmap="Blues", values_format="d", colorbar=False)
axes[0].set_title("Weighted Logistic Regression")

rf_disp = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=["H", "D", "A"]
)
rf_disp.plot(ax=axes[1], cmap="Greens", values_format="d", colorbar=False)
axes[1].set_title("Weighted Random Forest")

plt.suptitle(f"Confusion Matrices on Test Season ({latest_season})")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 16) CHOOSE BEST MODEL FOR FUTURE FIXTURE PREDICTIONS
# ------------------------------------------------------------
if best_model_name == "Weighted Random Forest":
    best_pipeline = weighted_random_forest_pipeline
else:
    best_pipeline = weighted_logistic_pipeline


# ------------------------------------------------------------
# 17) FIXTURE PREDICTION FUNCTION
# ------------------------------------------------------------
def predict_fixture(home_team, away_team, pipeline_to_use, pipeline_name):
    """
    Predicts the result probabilities for a new fixture using
    the latest available team stats and the chosen trained model.
    """
    current_season = matches["SourceFile"].iloc[-1]

    row = {
        "HomeTeam": home_team,
        "AwayTeam": away_team,

        "home_ppg": safe_avg(overall_stats[home_team]["points"], overall_stats[home_team]["played"], 1.30),
        "away_ppg": safe_avg(overall_stats[away_team]["points"], overall_stats[away_team]["played"], 1.30),

        "home_win_rate": safe_avg(overall_stats[home_team]["wins"], overall_stats[home_team]["played"], 0.33),
        "away_win_rate": safe_avg(overall_stats[away_team]["wins"], overall_stats[away_team]["played"], 0.33),

        "home_draw_rate": safe_avg(overall_stats[home_team]["draws"], overall_stats[home_team]["played"], 0.25),
        "away_draw_rate": safe_avg(overall_stats[away_team]["draws"], overall_stats[away_team]["played"], 0.25),

        "home_avg_gf": safe_avg(overall_stats[home_team]["gf"], overall_stats[home_team]["played"], 1.30),
        "away_avg_gf": safe_avg(overall_stats[away_team]["gf"], overall_stats[away_team]["played"], 1.30),

        "home_avg_ga": safe_avg(overall_stats[home_team]["ga"], overall_stats[home_team]["played"], 1.30),
        "away_avg_ga": safe_avg(overall_stats[away_team]["ga"], overall_stats[away_team]["played"], 1.30),

        "home_avg_shots": safe_avg(overall_stats[home_team]["shots"], overall_stats[home_team]["played"], 12.0),
        "away_avg_shots": safe_avg(overall_stats[away_team]["shots"], overall_stats[away_team]["played"], 12.0),

        "home_avg_sot": safe_avg(overall_stats[home_team]["sot"], overall_stats[home_team]["played"], 4.0),
        "away_avg_sot": safe_avg(overall_stats[away_team]["sot"], overall_stats[away_team]["played"], 4.0),

        "home_season_ppg": safe_avg(
            season_stats[current_season][home_team]["points"],
            season_stats[current_season][home_team]["played"],
            1.30
        ),
        "away_season_ppg": safe_avg(
            season_stats[current_season][away_team]["points"],
            season_stats[current_season][away_team]["played"],
            1.30
        ),

        "home_season_avg_gf": safe_avg(
            season_stats[current_season][home_team]["gf"],
            season_stats[current_season][home_team]["played"],
            1.30
        ),
        "away_season_avg_gf": safe_avg(
            season_stats[current_season][away_team]["gf"],
            season_stats[current_season][away_team]["played"],
            1.30
        ),

        "home_season_avg_ga": safe_avg(
            season_stats[current_season][home_team]["ga"],
            season_stats[current_season][home_team]["played"],
            1.30
        ),
        "away_season_avg_ga": safe_avg(
            season_stats[current_season][away_team]["ga"],
            season_stats[current_season][away_team]["played"],
            1.30
        ),

        "home_form_pts5": np.mean(recent_points[home_team]) if recent_points[home_team] else 1.20,
        "away_form_pts5": np.mean(recent_points[away_team]) if recent_points[away_team] else 1.20,

        "home_form_gd5": np.mean(recent_goal_diff[home_team]) if recent_goal_diff[home_team] else 0.0,
        "away_form_gd5": np.mean(recent_goal_diff[away_team]) if recent_goal_diff[away_team] else 0.0,
    }

    row["ppg_diff"] = row["home_ppg"] - row["away_ppg"]
    row["win_rate_diff"] = row["home_win_rate"] - row["away_win_rate"]
    row["draw_rate_diff"] = row["home_draw_rate"] - row["away_draw_rate"]
    row["avg_gf_diff"] = row["home_avg_gf"] - row["away_avg_gf"]
    row["avg_ga_diff"] = row["home_avg_ga"] - row["away_avg_ga"]
    row["avg_shots_diff"] = row["home_avg_shots"] - row["away_avg_shots"]
    row["avg_sot_diff"] = row["home_avg_sot"] - row["away_avg_sot"]
    row["season_ppg_diff"] = row["home_season_ppg"] - row["away_season_ppg"]
    row["season_avg_gf_diff"] = row["home_season_avg_gf"] - row["away_season_avg_gf"]
    row["season_avg_ga_diff"] = row["home_season_avg_ga"] - row["away_season_avg_ga"]
    row["form_pts5_diff"] = row["home_form_pts5"] - row["away_form_pts5"]
    row["form_gd5_diff"] = row["home_form_gd5"] - row["away_form_gd5"]

    fixture_df = pd.DataFrame([row])

    probs = pipeline_to_use.predict_proba(fixture_df)[0]
    classes = pipeline_to_use.named_steps["model"].classes_
    predicted_class = classes[np.argmax(probs)]
    result = dict(zip(classes, probs))

    print("\n" + "=" * 80)
    print(f"Prediction using {pipeline_name}")
    print(f"Fixture: {home_team} vs {away_team}")
    print(f"Predicted outcome: {predicted_class}")
    print("Probabilities:")
    for cls in classes:
        print(f"  {cls}: {result[cls]:.4f}")
    print("=" * 80)

    return predicted_class, result


# ------------------------------------------------------------
# 18) EXAMPLE FUTURE FIXTURE PREDICTIONS
# ------------------------------------------------------------
predict_fixture("Arsenal", "Chelsea", best_pipeline, best_model_name)
predict_fixture("Liverpool", "Man City", best_pipeline, best_model_name)