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
# 6. Predicts outcomes for new fixtures
#
# Libraries needed:
# pip install pandas numpy scikit-learn
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import glob
import os
from collections import defaultdict, deque

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ------------------------------------------------------------
# 1) LOAD ALL PREMIER LEAGUE CSV FILES
# ------------------------------------------------------------
# Change this path if your CSV files are stored somewhere else.
# Example:
# DATA_PATTERN = "data/*.csv"
DATA_PATTERN = "C:/Users/hensi/Downloads/match-predictor/E0*.csv"
# Find all matching CSV files
files = sorted(glob.glob(DATA_PATTERN))

if not files:
    raise FileNotFoundError(
        "No CSV files were found. Make sure your Premier League CSV files are in the same folder as this script."
    )

# Read each file and store it
all_dfs = []

for file in files:
    df = pd.read_csv(file)

    # Add source file name so we know which season each row came from
    df["SourceFile"] = os.path.basename(file)

    all_dfs.append(df)

# Combine all seasons into one DataFrame
matches = pd.concat(all_dfs, ignore_index=True, sort=False)


# ------------------------------------------------------------
# 2) BASIC CLEANING
# ------------------------------------------------------------
# These are the columns we need for this model
required_columns = [
    "Date",        # Match date
    "HomeTeam",    # Home team name
    "AwayTeam",    # Away team name
    "FTHG",        # Full-time home goals
    "FTAG",        # Full-time away goals
    "FTR",         # Full-time result (H/D/A)
    "HS",          # Home shots
    "AS",          # Away shots
    "HST",         # Home shots on target
    "AST",         # Away shots on target
    "SourceFile"   # Which CSV file/season it came from
]

# If any expected column is missing, create it with NaN
for col in required_columns:
    if col not in matches.columns:
        matches[col] = np.nan

# Keep only the columns we need
matches = matches[required_columns].copy()

# Convert Date to datetime
matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")

# Remove rows missing essential information
matches = matches.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])

# Keep only valid results
matches = matches[matches["FTR"].isin(["H", "D", "A"])].copy()

# Convert numeric columns safely
numeric_cols = ["FTHG", "FTAG", "HS", "AS", "HST", "AST"]

for col in numeric_cols:
    matches[col] = pd.to_numeric(matches[col], errors="coerce").fillna(0)

# Sort matches by date to ensure correct time order
matches = matches.sort_values("Date").reset_index(drop=True)


# ------------------------------------------------------------
# 3) HELPER FUNCTIONS
# ------------------------------------------------------------
def get_points(result, side):
    """
    Returns the number of points won by a team in a match.
    result: 'H', 'D', or 'A'
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
    Returns total / count if count > 0, otherwise returns a default value.
    This helps when a team has not yet played any games.
    """
    return total / count if count > 0 else default


# ------------------------------------------------------------
# 4) FEATURE ENGINEERING
# ------------------------------------------------------------
# Important:
# We create features using ONLY information available BEFORE each match.
# This prevents data leakage.

def build_pre_match_features(df):
    """
    Creates pre-match features for each game using past data only.

    Returns:
    - feature DataFrame
    - overall_stats
    - season_stats
    - recent_points
    - recent_goal_diff
    """

    # Overall team stats across all previous matches
    overall_stats = defaultdict(lambda: {
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0,
        "gf": 0,      # goals for
        "ga": 0,      # goals against
        "shots": 0,
        "sot": 0      # shots on target
    })

    # Season-specific stats
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

    # Last 5 match form trackers
    recent_points = defaultdict(lambda: deque(maxlen=5))
    recent_goal_diff = defaultdict(lambda: deque(maxlen=5))

    feature_rows = []

    # Process matches in time order
    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        season = row["SourceFile"]

        # Current stats BEFORE this match
        h_all = overall_stats[home]
        a_all = overall_stats[away]
        h_season = season_stats[season][home]
        a_season = season_stats[season][away]

        # Build features for this match
        feature_row = {
            "Date": row["Date"],
            "Season": season,
            "HomeTeam": home,
            "AwayTeam": away,
            "FTR": row["FTR"],

            # Overall performance
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

            # Current season performance
            "home_season_ppg": safe_avg(h_season["points"], h_season["played"], 1.30),
            "away_season_ppg": safe_avg(a_season["points"], a_season["played"], 1.30),

            "home_season_avg_gf": safe_avg(h_season["gf"], h_season["played"], 1.30),
            "away_season_avg_gf": safe_avg(a_season["gf"], a_season["played"], 1.30),

            "home_season_avg_ga": safe_avg(h_season["ga"], h_season["played"], 1.30),
            "away_season_avg_ga": safe_avg(a_season["ga"], a_season["played"], 1.30),

            # Last 5 match form
            "home_form_pts5": np.mean(recent_points[home]) if recent_points[home] else 1.20,
            "away_form_pts5": np.mean(recent_points[away]) if recent_points[away] else 1.20,

            "home_form_gd5": np.mean(recent_goal_diff[home]) if recent_goal_diff[home] else 0.0,
            "away_form_gd5": np.mean(recent_goal_diff[away]) if recent_goal_diff[away] else 0.0,
        }

        # Difference features: often useful in football prediction
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

        # Store features
        feature_rows.append(feature_row)

        # ------------------------------------------------------------
        # Update stats AFTER storing this match's pre-match features
        # ------------------------------------------------------------
        ftr = row["FTR"]
        fthg = row["FTHG"]
        ftag = row["FTAG"]
        hs = row["HS"]
        a_s = row["AS"]
        hst = row["HST"]
        ast = row["AST"]

        # Update home and away team stats
        for team, gf, ga, shots, sot, side in [
            (home, fthg, ftag, hs, hst, "home"),
            (away, ftag, fthg, a_s, ast, "away")
        ]:
            pts = get_points(ftr, side)

            # Update both overall and season-specific stats
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

        # Update recent form trackers
        recent_points[home].append(get_points(ftr, "home"))
        recent_points[away].append(get_points(ftr, "away"))

        recent_goal_diff[home].append(fthg - ftag)
        recent_goal_diff[away].append(ftag - fthg)

    return pd.DataFrame(feature_rows), overall_stats, season_stats, recent_points, recent_goal_diff


# Build features
feature_df, overall_stats, season_stats, recent_points, recent_goal_diff = build_pre_match_features(matches)


# ------------------------------------------------------------
# 5) TRAIN / TEST SPLIT
# ------------------------------------------------------------
# Use the most recent season as the test set
# This is better than a random split because football data is time-based.
latest_season = feature_df["Season"].iloc[-1]

train_df = feature_df[feature_df["Season"] != latest_season].copy()
test_df = feature_df[feature_df["Season"] == latest_season].copy()

# Separate features and target
X_train = train_df.drop(columns=["FTR", "Date", "Season"])
y_train = train_df["FTR"]

X_test = test_df.drop(columns=["FTR", "Date", "Season"])
y_test = test_df["FTR"]


# ------------------------------------------------------------
# 6) PREPROCESSING + MODEL
# ------------------------------------------------------------
# Identify numeric and categorical columns
numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = ["HomeTeam", "AwayTeam"]

# Preprocessing pipeline
# Numeric columns: fill missing values + scale
# Categorical columns: fill missing values + one-hot encode
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

# Logistic Regression model for multiclass classification
model = LogisticRegression(max_iter=3000)

# Full pipeline = preprocessing + model
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train the model
pipeline.fit(X_train, y_train)


# ------------------------------------------------------------
# 7) EVALUATE THE MODEL
# ------------------------------------------------------------
# Predict outcomes on the test set
test_predictions = pipeline.predict(X_test)
test_probabilities = pipeline.predict_proba(X_test)

print("=" * 70)
print(f"Training seasons: {train_df['Season'].nunique()}")
print(f"Test season: {latest_season}")
print(f"Train matches: {len(train_df)}")
print(f"Test matches: {len(test_df)}")
print("=" * 70)

# Accuracy
accuracy = accuracy_score(y_test, test_predictions)
print(f"Accuracy on held-out latest season: {accuracy:.3f}")
print()

# Detailed performance
print("Classification report:")
print(classification_report(y_test, test_predictions))


# ------------------------------------------------------------
# 8) SHOW SOME EXAMPLE PREDICTIONS FROM THE TEST SET
# ------------------------------------------------------------
# Convert probabilities into a readable table
proba_df = pd.DataFrame(
    test_probabilities,
    columns=pipeline.named_steps["model"].classes_
)

prediction_output = test_df[["Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
prediction_output["Predicted"] = test_predictions
prediction_output["Prob_H"] = proba_df.get("H", 0)
prediction_output["Prob_D"] = proba_df.get("D", 0)
prediction_output["Prob_A"] = proba_df.get("A", 0)

print("=" * 70)
print("Example predictions from the held-out latest season:")
print(prediction_output.head(15).to_string(index=False))
print("=" * 70)


# ------------------------------------------------------------
# 9) FUNCTION TO PREDICT A NEW FIXTURE
# ------------------------------------------------------------
def predict_fixture(home_team, away_team):
    """
    Predicts the result probabilities for a new fixture
    using the latest available team stats from the dataset.
    """

    current_season = matches["SourceFile"].iloc[-1]

    # Build one row of pre-match features
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

        "home_season_ppg": safe_avg(season_stats[current_season][home_team]["points"], season_stats[current_season][home_team]["played"], 1.30),
        "away_season_ppg": safe_avg(season_stats[current_season][away_team]["points"], season_stats[current_season][away_team]["played"], 1.30),

        "home_season_avg_gf": safe_avg(season_stats[current_season][home_team]["gf"], season_stats[current_season][home_team]["played"], 1.30),
        "away_season_avg_gf": safe_avg(season_stats[current_season][away_team]["gf"], season_stats[current_season][away_team]["played"], 1.30),

        "home_season_avg_ga": safe_avg(season_stats[current_season][home_team]["ga"], season_stats[current_season][home_team]["played"], 1.30),
        "away_season_avg_ga": safe_avg(season_stats[current_season][away_team]["ga"], season_stats[current_season][away_team]["played"], 1.30),

        "home_form_pts5": np.mean(recent_points[home_team]) if recent_points[home_team] else 1.20,
        "away_form_pts5": np.mean(recent_points[away_team]) if recent_points[away_team] else 1.20,

        "home_form_gd5": np.mean(recent_goal_diff[home_team]) if recent_goal_diff[home_team] else 0.0,
        "away_form_gd5": np.mean(recent_goal_diff[away_team]) if recent_goal_diff[away_team] else 0.0,
    }

    # Difference features
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

    # Convert into DataFrame for prediction
    fixture_df = pd.DataFrame([row])

    # Predict probabilities
    probs = pipeline.predict_proba(fixture_df)[0]
    classes = pipeline.named_steps["model"].classes_

    result = dict(zip(classes, probs))
    predicted_class = classes[np.argmax(probs)]

    print(f"\nPrediction for {home_team} vs {away_team}")
    print(f"Predicted outcome: {predicted_class}")
    print(f"Probabilities: {result}")

    return predicted_class, result


# ------------------------------------------------------------
# 10) EXAMPLE NEW FIXTURE PREDICTIONS
# ------------------------------------------------------------
# Change these team names to any teams in your dataset
predict_fixture("Arsenal", "Chelsea")
predict_fixture("Liverpool", "Man City")