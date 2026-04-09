# Premier League Match Outcome Prediction Model
# ------------------------------------------------------------
# This script:
# 1. Loads multiple Premier League CSV files
# 2. Cleans and combines the data
# 3. Builds pre-match features using only past information
# 4. Trains a machine learning model to predict:
#       H = Home Win
#       D = Draw
#       A = Away Win
# 5. Evaluates the model on the latest season
# 6. Visualises model performance with a confusion matrix
# 7. Predicts outcomes for new fixtures
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    f1_score,
    log_loss
)


# ------------------------------------------------------------
# 1) LOAD ALL PREMIER LEAGUE CSV FILES
# ------------------------------------------------------------
# Put this script in the same folder as the CSV files,
# or change the pattern to match your file location.
DATA_PATTERN = "C:/Users/hensi/Downloads/match-predictor/E0*.csv"

files = sorted(glob.glob(DATA_PATTERN))

if not files:
    raise FileNotFoundError(
        "No CSV files were found. Make sure your Premier League CSV files are in the same folder as this script."
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

# Create missing columns if needed
for col in required_columns:
    if col not in matches.columns:
        matches[col] = np.nan

matches = matches[required_columns].copy()

# Convert date column
matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")

# Drop rows with essential missing values
matches = matches.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])

# Keep only valid match outcomes
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
    Returns points won by a team.
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
    Safe division for averages.
    """
    return total / count if count > 0 else default


# ------------------------------------------------------------
# 4) FEATURE ENGINEERING
# ------------------------------------------------------------
def build_pre_match_features(df):
    """
    Creates pre-match features using ONLY information available
    before each match is played.
    """

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

        feature_row = {
            "Date": row["Date"],
            "Season": season,
            "HomeTeam": home,
            "AwayTeam": away,
            "FTR": row["FTR"],

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

            "home_season_ppg": safe_avg(h_season["points"], h_season["played"], 1.30),
            "away_season_ppg": safe_avg(a_season["points"], a_season["played"], 1.30),

            "home_season_avg_gf": safe_avg(h_season["gf"], h_season["played"], 1.30),
            "away_season_avg_gf": safe_avg(a_season["gf"], a_season["played"], 1.30),

            "home_season_avg_ga": safe_avg(h_season["ga"], h_season["played"], 1.30),
            "away_season_avg_ga": safe_avg(a_season["ga"], a_season["played"], 1.30),

            "home_form_pts5": np.mean(recent_points[home]) if recent_points[home] else 1.20,
            "away_form_pts5": np.mean(recent_points[away]) if recent_points[away] else 1.20,

            "home_form_gd5": np.mean(recent_goal_diff[home]) if recent_goal_diff[home] else 0.0,
            "away_form_gd5": np.mean(recent_goal_diff[away]) if recent_goal_diff[away] else 0.0,
        }

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

        # Update stats AFTER creating pre-match features
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
# Hold out the most recent season for testing
latest_season = feature_df["Season"].iloc[-1]

train_df = feature_df[feature_df["Season"] != latest_season].copy()
test_df = feature_df[feature_df["Season"] == latest_season].copy()

X_train = train_df.drop(columns=["FTR", "Date", "Season"])
y_train = train_df["FTR"]

X_test = test_df.drop(columns=["FTR", "Date", "Season"])
y_test = test_df["FTR"]


# ------------------------------------------------------------
# 6) PREPROCESSING + MODEL
# ------------------------------------------------------------
numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = ["HomeTeam", "AwayTeam"]

preprocessor = ColumnTransformer(
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

model = LogisticRegression(max_iter=3000)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)


# ------------------------------------------------------------
# 7) EVALUATE THE MODEL
# ------------------------------------------------------------
test_predictions = pipeline.predict(X_test)
test_probabilities = pipeline.predict_proba(X_test)
class_labels = pipeline.named_steps["model"].classes_

print("=" * 80)
print("MODEL EVALUATION")
print("=" * 80)
print(f"Training seasons: {train_df['Season'].nunique()}")
print(f"Test season: {latest_season}")
print(f"Train matches: {len(train_df)}")
print(f"Test matches: {len(test_df)}")
print("-" * 80)

# Main evaluation metrics
accuracy = accuracy_score(y_test, test_predictions)
balanced_acc = balanced_accuracy_score(y_test, test_predictions)
macro_f1 = f1_score(y_test, test_predictions, average="macro")
weighted_f1 = f1_score(y_test, test_predictions, average="weighted")
multiclass_logloss = log_loss(y_test, test_probabilities, labels=class_labels)

print(f"Accuracy:           {accuracy:.4f}")
print(f"Balanced Accuracy:  {balanced_acc:.4f}")
print(f"Macro F1 Score:     {macro_f1:.4f}")
print(f"Weighted F1 Score:  {weighted_f1:.4f}")
print(f"Log Loss:           {multiclass_logloss:.4f}")
print("-" * 80)

print("Classification Report:")
print(classification_report(y_test, test_predictions, digits=4))


# ------------------------------------------------------------
# 8) ACTUAL VS PREDICTED COUNTS
# ------------------------------------------------------------
actual_counts = y_test.value_counts().sort_index()
pred_counts = pd.Series(test_predictions).value_counts().sort_index()

count_summary = pd.DataFrame({
    "Actual_Count": actual_counts,
    "Predicted_Count": pred_counts
}).fillna(0).astype(int)

print("-" * 80)
print("Actual vs Predicted Outcome Counts:")
print(count_summary)
print("-" * 80)


# ------------------------------------------------------------
# 9) CONFUSION MATRIX
# ------------------------------------------------------------
cm = confusion_matrix(y_test, test_predictions, labels=["H", "D", "A"])

print("Confusion Matrix (rows = actual, columns = predicted):")
cm_df = pd.DataFrame(
    cm,
    index=["Actual_H", "Actual_D", "Actual_A"],
    columns=["Pred_H", "Pred_D", "Pred_A"]
)
print(cm_df)
print("-" * 80)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["H", "D", "A"])
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix - Test Season ({latest_season})")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10) PROBABILITY TABLE FOR TEST PREDICTIONS
# ------------------------------------------------------------
proba_df = pd.DataFrame(test_probabilities, columns=class_labels)

prediction_output = test_df[["Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
prediction_output["Predicted"] = test_predictions
prediction_output["Correct"] = prediction_output["FTR"] == prediction_output["Predicted"]

# Add per-class probabilities safely
prediction_output["Prob_H"] = proba_df["H"] if "H" in proba_df.columns else 0.0
prediction_output["Prob_D"] = proba_df["D"] if "D" in proba_df.columns else 0.0
prediction_output["Prob_A"] = proba_df["A"] if "A" in proba_df.columns else 0.0

# Confidence = highest class probability for that match
prediction_output["Confidence"] = prediction_output[["Prob_H", "Prob_D", "Prob_A"]].max(axis=1)

print("Sample Predictions:")
print(
    prediction_output[
        ["Date", "HomeTeam", "AwayTeam", "FTR", "Predicted", "Correct", "Prob_H", "Prob_D", "Prob_A", "Confidence"]
    ]
    .head(15)
    .to_string(index=False)
)
print("-" * 80)


# ------------------------------------------------------------
# 11) MOST CONFIDENT PREDICTIONS
# ------------------------------------------------------------
most_confident = prediction_output.sort_values("Confidence", ascending=False).head(10)

print("Top 10 Most Confident Predictions:")
print(
    most_confident[
        ["Date", "HomeTeam", "AwayTeam", "FTR", "Predicted", "Correct", "Prob_H", "Prob_D", "Prob_A", "Confidence"]
    ].to_string(index=False)
)
print("-" * 80)


# ------------------------------------------------------------
# 12) CLASS-SPECIFIC ACCURACY
# ------------------------------------------------------------
# This shows how often each actual class was predicted correctly.
class_accuracy_rows = []

for label in ["H", "D", "A"]:
    actual_mask = y_test == label
    total_actual = actual_mask.sum()

    if total_actual > 0:
        correct_actual = (test_predictions[actual_mask] == label).sum()
        class_acc = correct_actual / total_actual
    else:
        correct_actual = 0
        class_acc = np.nan

    class_accuracy_rows.append({
        "Class": label,
        "Actual_Count": int(total_actual),
        "Correctly_Predicted": int(correct_actual),
        "Class_Accuracy": round(class_acc, 4) if not np.isnan(class_acc) else np.nan
    })

class_accuracy_df = pd.DataFrame(class_accuracy_rows)

print("Class-Specific Accuracy:")
print(class_accuracy_df.to_string(index=False))
print("=" * 80)


# ------------------------------------------------------------
# 13) FUNCTION TO PREDICT A NEW FIXTURE
# ------------------------------------------------------------
def predict_fixture(home_team, away_team):
    """
    Predict result probabilities for a new fixture using
    the latest available team stats.
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

    probs = pipeline.predict_proba(fixture_df)[0]
    classes = pipeline.named_steps["model"].classes_

    result = dict(zip(classes, probs))
    predicted_class = classes[np.argmax(probs)]

    print(f"\nPrediction for {home_team} vs {away_team}")
    print(f"Predicted outcome: {predicted_class}")
    print(f"Probabilities:")
    for cls in classes:
        print(f"  {cls}: {result[cls]:.4f}")

    return predicted_class, result


# ------------------------------------------------------------
# 14) EXAMPLE NEW FIXTURE PREDICTIONS
# ------------------------------------------------------------
# Change to any teams that exist in your dataset
predict_fixture("Arsenal", "Chelsea")
predict_fixture("Liverpool", "Man City")