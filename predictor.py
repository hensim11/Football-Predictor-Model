# Premier League Match Outcome Prediction Model - Improved Feature Engineering
# ---------------------------------------------------------------------------
# This version improves the feature engineering to better reflect team strength.
#
# Main improvements:
# 1. Recent form features (last 3 and last 5 matches)
# 2. Home-only and away-only performance features
# 3. ELO ratings for team strength
# 4. Attack/defence indicators from goals and shots
# 5. Current-season table strength before each match
# 6. Rest days
# 7. Many difference features between home and away teams
#
# Models compared:
# - Weighted Logistic Regression
# - Weighted Random Forest
#
# Libraries needed:
# pip install pandas numpy scikit-learn matplotlib
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 1) LOAD ALL PREMIER LEAGUE CSV FILES
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2) BASIC CLEANING
# ---------------------------------------------------------------------------
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

for col in required_columns:
    if col not in matches.columns:
        matches[col] = np.nan

matches = matches[required_columns].copy()

matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")
matches = matches.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
matches = matches[matches["FTR"].isin(["H", "D", "A"])].copy()

numeric_cols = ["FTHG", "FTAG", "HS", "AS", "HST", "AST"]
for col in numeric_cols:
    matches[col] = pd.to_numeric(matches[col], errors="coerce").fillna(0)

matches = matches.sort_values(["Date", "SourceFile"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3) HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def get_points(result, side):
    """
    Returns league points won by a team from a match result.
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
    Safe average with fallback default for teams with no prior matches.
    """
    return total / count if count > 0 else default


def deque_mean(dq, default):
    """
    Mean of a deque if it has values, otherwise a default.
    """
    return float(np.mean(dq)) if len(dq) > 0 else default


def days_since(last_date, current_date, default=7.0):
    """
    Returns number of days since previous match for a team.
    Uses default if team has no prior match date.
    """
    if last_date is None:
        return default
    diff = (current_date - last_date).days
    if pd.isna(diff):
        return default
    return max(float(diff), 0.0)


def elo_expected(rating_a, rating_b, home_adv=0):
    """
    Expected score for team A against team B using ELO.
    """
    return 1 / (1 + 10 ** (((rating_b) - (rating_a + home_adv)) / 400))


def elo_actual_score(result, side):
    """
    Converts match result to ELO actual score.
    Win = 1, Draw = 0.5, Loss = 0
    """
    if result == "D":
        return 0.5
    if result == "H" and side == "home":
        return 1.0
    if result == "A" and side == "away":
        return 1.0
    return 0.0


def league_position_proxy(points, gd, gf, team_name):
    """
    Used only for sorting current season table before each match.
    Higher points > higher gd > higher gf.
    """
    return (points, gd, gf, team_name)


# ---------------------------------------------------------------------------
# 4) FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def build_pre_match_features(df):
    """
    Builds improved pre-match features using only information available
    before each match is played.
    """

    # -----------------------------
    # Long-run overall stats
    # -----------------------------
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

    # -----------------------------
    # Home-only / away-only stats
    # -----------------------------
    home_stats = defaultdict(lambda: {
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

    away_stats = defaultdict(lambda: {
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

    # -----------------------------
    # Current-season stats
    # -----------------------------
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

    # -----------------------------
    # Recent form trackers
    # -----------------------------
    recent_points_3 = defaultdict(lambda: deque(maxlen=3))
    recent_points_5 = defaultdict(lambda: deque(maxlen=5))

    recent_gf_3 = defaultdict(lambda: deque(maxlen=3))
    recent_gf_5 = defaultdict(lambda: deque(maxlen=5))

    recent_ga_3 = defaultdict(lambda: deque(maxlen=3))
    recent_ga_5 = defaultdict(lambda: deque(maxlen=5))

    recent_gd_3 = defaultdict(lambda: deque(maxlen=3))
    recent_gd_5 = defaultdict(lambda: deque(maxlen=5))

    recent_shots_3 = defaultdict(lambda: deque(maxlen=3))
    recent_shots_5 = defaultdict(lambda: deque(maxlen=5))

    recent_sot_3 = defaultdict(lambda: deque(maxlen=3))
    recent_sot_5 = defaultdict(lambda: deque(maxlen=5))

    # Venue-specific recent form
    recent_home_points_3 = defaultdict(lambda: deque(maxlen=3))
    recent_home_points_5 = defaultdict(lambda: deque(maxlen=5))
    recent_away_points_3 = defaultdict(lambda: deque(maxlen=3))
    recent_away_points_5 = defaultdict(lambda: deque(maxlen=5))

    # Rest / scheduling
    last_match_date = defaultdict(lambda: None)

    # ELO ratings
    elo_rating = defaultdict(lambda: 1500.0)
    home_elo_rating = defaultdict(lambda: 1500.0)
    away_elo_rating = defaultdict(lambda: 1500.0)

    # Hyperparameters for ELO
    K = 20
    HOME_ADV = 60

    feature_rows = []

    # Process matches in strict chronological order
    for _, row in df.iterrows():
        date = row["Date"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        season = row["SourceFile"]

        h_all = overall_stats[home]
        a_all = overall_stats[away]

        h_home = home_stats[home]
        a_away = away_stats[away]

        h_season = season_stats[season][home]
        a_season = season_stats[season][away]

        # -------------------------------------------------------------------
        # Current table positions before the match
        # -------------------------------------------------------------------
        season_table = []
        season_team_dict = season_stats[season]

        for team_name, stats in season_team_dict.items():
            points = stats["points"]
            gd = stats["gf"] - stats["ga"]
            gf = stats["gf"]
            season_table.append((team_name, points, gd, gf))

        # Sort descending by points, gd, gf
        season_table_sorted = sorted(
            season_table,
            key=lambda x: (x[1], x[2], x[3], x[0]),
            reverse=True
        )

        table_position_lookup = {}
        for pos, (team_name, _, _, _) in enumerate(season_table_sorted, start=1):
            table_position_lookup[team_name] = pos

        home_table_pos = table_position_lookup.get(home, 20)
        away_table_pos = table_position_lookup.get(away, 20)

        # -------------------------------------------------------------------
        # Rest days
        # -------------------------------------------------------------------
        home_rest_days = days_since(last_match_date[home], date, default=7.0)
        away_rest_days = days_since(last_match_date[away], date, default=7.0)

        # -------------------------------------------------------------------
        # ELO features before the match
        # -------------------------------------------------------------------
        home_elo = elo_rating[home]
        away_elo = elo_rating[away]

        home_home_elo = home_elo_rating[home]
        away_away_elo = away_elo_rating[away]

        elo_exp_home = elo_expected(home_elo, away_elo, home_adv=HOME_ADV)
        elo_exp_away = 1 - elo_exp_home

        venue_elo_exp_home = elo_expected(home_home_elo, away_away_elo, home_adv=HOME_ADV)
        venue_elo_exp_away = 1 - venue_elo_exp_home

        # -------------------------------------------------------------------
        # Pre-match features
        # -------------------------------------------------------------------
        feature_row = {
            "Date": date,
            "Season": season,
            "HomeTeam": home,
            "AwayTeam": away,
            "FTR": row["FTR"],

            # -------------------------
            # Overall long-run features
            # -------------------------
            "home_ppg_all": safe_avg(h_all["points"], h_all["played"], 1.35),
            "away_ppg_all": safe_avg(a_all["points"], a_all["played"], 1.10),

            "home_win_rate_all": safe_avg(h_all["wins"], h_all["played"], 0.35),
            "away_win_rate_all": safe_avg(a_all["wins"], a_all["played"], 0.30),

            "home_draw_rate_all": safe_avg(h_all["draws"], h_all["played"], 0.25),
            "away_draw_rate_all": safe_avg(a_all["draws"], a_all["played"], 0.25),

            "home_avg_gf_all": safe_avg(h_all["gf"], h_all["played"], 1.40),
            "away_avg_gf_all": safe_avg(a_all["gf"], a_all["played"], 1.10),

            "home_avg_ga_all": safe_avg(h_all["ga"], h_all["played"], 1.20),
            "away_avg_ga_all": safe_avg(a_all["ga"], a_all["played"], 1.30),

            "home_avg_shots_all": safe_avg(h_all["shots"], h_all["played"], 12.0),
            "away_avg_shots_all": safe_avg(a_all["shots"], a_all["played"], 11.0),

            "home_avg_sot_all": safe_avg(h_all["sot"], h_all["played"], 4.5),
            "away_avg_sot_all": safe_avg(a_all["sot"], a_all["played"], 4.0),

            # -------------------------
            # Venue-specific features
            # -------------------------
            "home_ppg_home": safe_avg(h_home["points"], h_home["played"], 1.50),
            "away_ppg_away": safe_avg(a_away["points"], a_away["played"], 0.95),

            "home_win_rate_home": safe_avg(h_home["wins"], h_home["played"], 0.45),
            "away_win_rate_away": safe_avg(a_away["wins"], a_away["played"], 0.25),

            "home_avg_gf_home": safe_avg(h_home["gf"], h_home["played"], 1.55),
            "away_avg_gf_away": safe_avg(a_away["gf"], a_away["played"], 1.00),

            "home_avg_ga_home": safe_avg(h_home["ga"], h_home["played"], 1.10),
            "away_avg_ga_away": safe_avg(a_away["ga"], a_away["played"], 1.45),

            "home_avg_shots_home": safe_avg(h_home["shots"], h_home["played"], 13.0),
            "away_avg_shots_away": safe_avg(a_away["shots"], a_away["played"], 10.5),

            "home_avg_sot_home": safe_avg(h_home["sot"], h_home["played"], 4.8),
            "away_avg_sot_away": safe_avg(a_away["sot"], a_away["played"], 3.8),

            # -------------------------
            # Current-season features
            # -------------------------
            "home_season_ppg": safe_avg(h_season["points"], h_season["played"], 1.35),
            "away_season_ppg": safe_avg(a_season["points"], a_season["played"], 1.10),

            "home_season_gf_pg": safe_avg(h_season["gf"], h_season["played"], 1.40),
            "away_season_gf_pg": safe_avg(a_season["gf"], a_season["played"], 1.10),

            "home_season_ga_pg": safe_avg(h_season["ga"], h_season["played"], 1.20),
            "away_season_ga_pg": safe_avg(a_season["ga"], a_season["played"], 1.30),

            "home_season_gd_pg": safe_avg(h_season["gf"] - h_season["ga"], h_season["played"], 0.10),
            "away_season_gd_pg": safe_avg(a_season["gf"] - a_season["ga"], a_season["played"], -0.10),

            "home_season_shots_pg": safe_avg(h_season["shots"], h_season["played"], 12.0),
            "away_season_shots_pg": safe_avg(a_season["shots"], a_season["played"], 11.0),

            "home_season_sot_pg": safe_avg(h_season["sot"], h_season["played"], 4.5),
            "away_season_sot_pg": safe_avg(a_season["sot"], a_season["played"], 4.0),

            "home_table_pos": home_table_pos,
            "away_table_pos": away_table_pos,

            # -------------------------
            # Recent form - last 3
            # -------------------------
            "home_form_pts_3": deque_mean(recent_points_3[home], 1.30),
            "away_form_pts_3": deque_mean(recent_points_3[away], 1.10),

            "home_form_gf_3": deque_mean(recent_gf_3[home], 1.40),
            "away_form_gf_3": deque_mean(recent_gf_3[away], 1.10),

            "home_form_ga_3": deque_mean(recent_ga_3[home], 1.20),
            "away_form_ga_3": deque_mean(recent_ga_3[away], 1.30),

            "home_form_gd_3": deque_mean(recent_gd_3[home], 0.10),
            "away_form_gd_3": deque_mean(recent_gd_3[away], -0.10),

            "home_form_shots_3": deque_mean(recent_shots_3[home], 12.0),
            "away_form_shots_3": deque_mean(recent_shots_3[away], 11.0),

            "home_form_sot_3": deque_mean(recent_sot_3[home], 4.5),
            "away_form_sot_3": deque_mean(recent_sot_3[away], 4.0),

            # -------------------------
            # Recent form - last 5
            # -------------------------
            "home_form_pts_5": deque_mean(recent_points_5[home], 1.30),
            "away_form_pts_5": deque_mean(recent_points_5[away], 1.10),

            "home_form_gf_5": deque_mean(recent_gf_5[home], 1.40),
            "away_form_gf_5": deque_mean(recent_gf_5[away], 1.10),

            "home_form_ga_5": deque_mean(recent_ga_5[home], 1.20),
            "away_form_ga_5": deque_mean(recent_ga_5[away], 1.30),

            "home_form_gd_5": deque_mean(recent_gd_5[home], 0.10),
            "away_form_gd_5": deque_mean(recent_gd_5[away], -0.10),

            "home_form_shots_5": deque_mean(recent_shots_5[home], 12.0),
            "away_form_shots_5": deque_mean(recent_shots_5[away], 11.0),

            "home_form_sot_5": deque_mean(recent_sot_5[home], 4.5),
            "away_form_sot_5": deque_mean(recent_sot_5[away], 4.0),

            # -------------------------
            # Venue-specific recent form
            # -------------------------
            "home_home_form_pts_3": deque_mean(recent_home_points_3[home], 1.45),
            "home_home_form_pts_5": deque_mean(recent_home_points_5[home], 1.45),

            "away_away_form_pts_3": deque_mean(recent_away_points_3[away], 1.00),
            "away_away_form_pts_5": deque_mean(recent_away_points_5[away], 1.00),

            # -------------------------
            # ELO features
            # -------------------------
            "home_elo": home_elo,
            "away_elo": away_elo,
            "home_home_elo": home_home_elo,
            "away_away_elo": away_away_elo,
            "home_elo_expected": elo_exp_home,
            "away_elo_expected": elo_exp_away,
            "home_venue_elo_expected": venue_elo_exp_home,
            "away_venue_elo_expected": venue_elo_exp_away,

            # -------------------------
            # Rest
            # -------------------------
            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days,
        }

        # -------------------------------------------------------------------
        # Difference features
        # -------------------------------------------------------------------
        diff_pairs = [
            ("ppg_all_diff", "home_ppg_all", "away_ppg_all"),
            ("win_rate_all_diff", "home_win_rate_all", "away_win_rate_all"),
            ("draw_rate_all_diff", "home_draw_rate_all", "away_draw_rate_all"),
            ("avg_gf_all_diff", "home_avg_gf_all", "away_avg_gf_all"),
            ("avg_ga_all_diff", "home_avg_ga_all", "away_avg_ga_all"),
            ("avg_shots_all_diff", "home_avg_shots_all", "away_avg_shots_all"),
            ("avg_sot_all_diff", "home_avg_sot_all", "away_avg_sot_all"),

            ("ppg_venue_diff", "home_ppg_home", "away_ppg_away"),
            ("win_rate_venue_diff", "home_win_rate_home", "away_win_rate_away"),
            ("gf_venue_diff", "home_avg_gf_home", "away_avg_gf_away"),
            ("ga_venue_diff", "home_avg_ga_home", "away_avg_ga_away"),
            ("shots_venue_diff", "home_avg_shots_home", "away_avg_shots_away"),
            ("sot_venue_diff", "home_avg_sot_home", "away_avg_sot_away"),

            ("season_ppg_diff", "home_season_ppg", "away_season_ppg"),
            ("season_gf_pg_diff", "home_season_gf_pg", "away_season_gf_pg"),
            ("season_ga_pg_diff", "home_season_ga_pg", "away_season_ga_pg"),
            ("season_gd_pg_diff", "home_season_gd_pg", "away_season_gd_pg"),
            ("season_shots_pg_diff", "home_season_shots_pg", "away_season_shots_pg"),
            ("season_sot_pg_diff", "home_season_sot_pg", "away_season_sot_pg"),

            ("form_pts_3_diff", "home_form_pts_3", "away_form_pts_3"),
            ("form_pts_5_diff", "home_form_pts_5", "away_form_pts_5"),

            ("form_gf_3_diff", "home_form_gf_3", "away_form_gf_3"),
            ("form_gf_5_diff", "home_form_gf_5", "away_form_gf_5"),

            ("form_ga_3_diff", "home_form_ga_3", "away_form_ga_3"),
            ("form_ga_5_diff", "home_form_ga_5", "away_form_ga_5"),

            ("form_gd_3_diff", "home_form_gd_3", "away_form_gd_3"),
            ("form_gd_5_diff", "home_form_gd_5", "away_form_gd_5"),

            ("form_shots_3_diff", "home_form_shots_3", "away_form_shots_3"),
            ("form_shots_5_diff", "home_form_shots_5", "away_form_shots_5"),

            ("form_sot_3_diff", "home_form_sot_3", "away_form_sot_3"),
            ("form_sot_5_diff", "home_form_sot_5", "away_form_sot_5"),

            ("home_away_form_pts_3_diff", "home_home_form_pts_3", "away_away_form_pts_3"),
            ("home_away_form_pts_5_diff", "home_home_form_pts_5", "away_away_form_pts_5"),

            ("elo_diff", "home_elo", "away_elo"),
            ("venue_elo_diff", "home_home_elo", "away_away_elo"),
            ("elo_expected_diff", "home_elo_expected", "away_elo_expected"),
            ("venue_elo_expected_diff", "home_venue_elo_expected", "away_venue_elo_expected"),

            ("rest_days_diff", "home_rest_days", "away_rest_days"),
        ]

        for diff_name, left, right in diff_pairs:
            feature_row[diff_name] = feature_row[left] - feature_row[right]

        # Lower table position number is better, so we flip this
        feature_row["table_pos_diff"] = feature_row["away_table_pos"] - feature_row["home_table_pos"]

        feature_rows.append(feature_row)

        # -------------------------------------------------------------------
        # Update trackers AFTER creating features (to avoid leakage)
        # -------------------------------------------------------------------
        ftr = row["FTR"]
        fthg = row["FTHG"]
        ftag = row["FTAG"]
        hs = row["HS"]
        a_s = row["AS"]
        hst = row["HST"]
        ast = row["AST"]

        home_pts = get_points(ftr, "home")
        away_pts = get_points(ftr, "away")

        # Update overall + season stats
        for team, gf, ga, shots, sot, pts in [
            (home, fthg, ftag, hs, hst, home_pts),
            (away, ftag, fthg, a_s, ast, away_pts)
        ]:
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

        # Update venue-only stats
        for stat_dict, gf, ga, shots, sot, pts in [
            (home_stats[home], fthg, ftag, hs, hst, home_pts),
            (away_stats[away], ftag, fthg, a_s, ast, away_pts)
        ]:
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

        # Update recent form
        recent_points_3[home].append(home_pts)
        recent_points_3[away].append(away_pts)
        recent_points_5[home].append(home_pts)
        recent_points_5[away].append(away_pts)

        recent_gf_3[home].append(fthg)
        recent_gf_3[away].append(ftag)
        recent_gf_5[home].append(fthg)
        recent_gf_5[away].append(ftag)

        recent_ga_3[home].append(ftag)
        recent_ga_3[away].append(fthg)
        recent_ga_5[home].append(ftag)
        recent_ga_5[away].append(fthg)

        recent_gd_3[home].append(fthg - ftag)
        recent_gd_3[away].append(ftag - fthg)
        recent_gd_5[home].append(fthg - ftag)
        recent_gd_5[away].append(ftag - fthg)

        recent_shots_3[home].append(hs)
        recent_shots_3[away].append(a_s)
        recent_shots_5[home].append(hs)
        recent_shots_5[away].append(a_s)

        recent_sot_3[home].append(hst)
        recent_sot_3[away].append(ast)
        recent_sot_5[home].append(hst)
        recent_sot_5[away].append(ast)

        # Venue-specific recent form
        recent_home_points_3[home].append(home_pts)
        recent_home_points_5[home].append(home_pts)

        recent_away_points_3[away].append(away_pts)
        recent_away_points_5[away].append(away_pts)

        # Update rest dates
        last_match_date[home] = date
        last_match_date[away] = date

        # Update ELO
        home_expected = elo_exp_home
        away_expected = elo_exp_away

        home_actual = elo_actual_score(ftr, "home")
        away_actual = elo_actual_score(ftr, "away")

        elo_rating[home] += K * (home_actual - home_expected)
        elo_rating[away] += K * (away_actual - away_expected)

        home_home_expected = venue_elo_exp_home
        away_away_expected = venue_elo_exp_away

        home_elo_rating[home] += K * (home_actual - home_home_expected)
        away_elo_rating[away] += K * (away_actual - away_away_expected)

    return pd.DataFrame(feature_rows)


feature_df = build_pre_match_features(matches)


# ---------------------------------------------------------------------------
# 5) TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
latest_season = feature_df["Season"].iloc[-1]

train_df = feature_df[feature_df["Season"] != latest_season].copy()
test_df = feature_df[feature_df["Season"] == latest_season].copy()

X_train = train_df.drop(columns=["FTR", "Date", "Season"])
y_train = train_df["FTR"]

X_test = test_df.drop(columns=["FTR", "Date", "Season"])
y_test = test_df["FTR"]


# ---------------------------------------------------------------------------
# 6) CLASS IMBALANCE
# ---------------------------------------------------------------------------
print("=" * 80)
print("CLASS DISTRIBUTION IN TRAINING DATA")
print("=" * 80)
print(y_train.value_counts().sort_index())
print()

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


# ---------------------------------------------------------------------------
# 7) PREPROCESSING
# ---------------------------------------------------------------------------
numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = ["HomeTeam", "AwayTeam"]

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


# ---------------------------------------------------------------------------
# 8) BUILD MODELS
# ---------------------------------------------------------------------------
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
        n_estimators=600,
        max_depth=14,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    ))
])


# ---------------------------------------------------------------------------
# 9) TRAIN MODELS
# ---------------------------------------------------------------------------
print("TRAINING WEIGHTED LOGISTIC REGRESSION...")
weighted_logistic_pipeline.fit(X_train, y_train)

print("TRAINING WEIGHTED RANDOM FOREST...")
weighted_random_forest_pipeline.fit(X_train, y_train)

print("TRAINING COMPLETE.")
print("=" * 80)


# ---------------------------------------------------------------------------
# 10) EVALUATION FUNCTION
# ---------------------------------------------------------------------------
def evaluate_model(model_name, pipeline, X_test, y_test):
    """
    Evaluates a trained model and returns useful outputs.
    """
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    model_classes = pipeline.named_steps["model"].classes_

    accuracy = accuracy_score(y_test, predictions)
    balanced_acc = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")
    multiclass_logloss = log_loss(y_test, probabilities, labels=model_classes)

    precision, recall, f1_vals, support = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=["H", "D", "A"],
        zero_division=0
    )

    class_breakdown = pd.DataFrame({
        "Class": ["H", "D", "A"],
        "Precision": precision,
        "Recall": recall,
        "F1": f1_vals,
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


# ---------------------------------------------------------------------------
# 11) EVALUATE BOTH MODELS
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 12) COMPARE MODEL PERFORMANCE
# ---------------------------------------------------------------------------
comparison_df = pd.DataFrame([log_metrics, rf_metrics])

print("\n" + "=" * 80)
print("WEIGHTED MODEL COMPARISON")
print("=" * 80)
print(comparison_df.to_string(index=False))
print("=" * 80)

best_model_row = comparison_df.sort_values(
    by=["Balanced_Accuracy", "Macro_F1", "Accuracy"],
    ascending=False
).iloc[0]

best_model_name = best_model_row["Model"]
print(f"Best weighted model by this comparison: {best_model_name}")
print("=" * 80)


# ---------------------------------------------------------------------------
# 13) ACTUAL VS PREDICTED COUNTS
# ---------------------------------------------------------------------------
def print_actual_vs_predicted_counts(model_name, y_true, y_pred):
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


# ---------------------------------------------------------------------------
# 14) BUILD RESULTS TABLES
# ---------------------------------------------------------------------------
def build_prediction_output(test_df, predictions, probabilities, model_classes, model_name):
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


# ---------------------------------------------------------------------------
# 15) CONFUSION MATRICES
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 16) CHOOSE BEST MODEL FOR FUTURE FIXTURE PREDICTIONS
# ---------------------------------------------------------------------------
if best_model_name == "Weighted Random Forest":
    best_pipeline = weighted_random_forest_pipeline
else:
    best_pipeline = weighted_logistic_pipeline


# ---------------------------------------------------------------------------
# 17) REBUILD TRACKERS ON FULL DATA FOR FUTURE FIXTURE PREDICTION
# ---------------------------------------------------------------------------
def rebuild_full_state_for_prediction(df):
    """
    Rebuilds the same state trackers on all available data so we can
    predict a future fixture using the most up-to-date information.
    """

    overall_stats = defaultdict(lambda: {
        "played": 0, "wins": 0, "draws": 0, "losses": 0,
        "points": 0, "gf": 0, "ga": 0, "shots": 0, "sot": 0
    })
    home_stats = defaultdict(lambda: {
        "played": 0, "wins": 0, "draws": 0, "losses": 0,
        "points": 0, "gf": 0, "ga": 0, "shots": 0, "sot": 0
    })
    away_stats = defaultdict(lambda: {
        "played": 0, "wins": 0, "draws": 0, "losses": 0,
        "points": 0, "gf": 0, "ga": 0, "shots": 0, "sot": 0
    })
    season_stats = defaultdict(lambda: defaultdict(lambda: {
        "played": 0, "wins": 0, "draws": 0, "losses": 0,
        "points": 0, "gf": 0, "ga": 0, "shots": 0, "sot": 0
    }))

    recent_points_3 = defaultdict(lambda: deque(maxlen=3))
    recent_points_5 = defaultdict(lambda: deque(maxlen=5))
    recent_gf_3 = defaultdict(lambda: deque(maxlen=3))
    recent_gf_5 = defaultdict(lambda: deque(maxlen=5))
    recent_ga_3 = defaultdict(lambda: deque(maxlen=3))
    recent_ga_5 = defaultdict(lambda: deque(maxlen=5))
    recent_gd_3 = defaultdict(lambda: deque(maxlen=3))
    recent_gd_5 = defaultdict(lambda: deque(maxlen=5))
    recent_shots_3 = defaultdict(lambda: deque(maxlen=3))
    recent_shots_5 = defaultdict(lambda: deque(maxlen=5))
    recent_sot_3 = defaultdict(lambda: deque(maxlen=3))
    recent_sot_5 = defaultdict(lambda: deque(maxlen=5))
    recent_home_points_3 = defaultdict(lambda: deque(maxlen=3))
    recent_home_points_5 = defaultdict(lambda: deque(maxlen=5))
    recent_away_points_3 = defaultdict(lambda: deque(maxlen=3))
    recent_away_points_5 = defaultdict(lambda: deque(maxlen=5))
    last_match_date = defaultdict(lambda: None)

    elo_rating = defaultdict(lambda: 1500.0)
    home_elo_rating = defaultdict(lambda: 1500.0)
    away_elo_rating = defaultdict(lambda: 1500.0)

    K = 20
    HOME_ADV = 60

    for _, row in df.iterrows():
        date = row["Date"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        season = row["SourceFile"]

        ftr = row["FTR"]
        fthg = row["FTHG"]
        ftag = row["FTAG"]
        hs = row["HS"]
        a_s = row["AS"]
        hst = row["HST"]
        ast = row["AST"]

        home_pts = get_points(ftr, "home")
        away_pts = get_points(ftr, "away")

        for team, gf, ga, shots, sot, pts in [
            (home, fthg, ftag, hs, hst, home_pts),
            (away, ftag, fthg, a_s, ast, away_pts)
        ]:
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

        for stat_dict, gf, ga, shots, sot, pts in [
            (home_stats[home], fthg, ftag, hs, hst, home_pts),
            (away_stats[away], ftag, fthg, a_s, ast, away_pts)
        ]:
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

        recent_points_3[home].append(home_pts)
        recent_points_3[away].append(away_pts)
        recent_points_5[home].append(home_pts)
        recent_points_5[away].append(away_pts)

        recent_gf_3[home].append(fthg)
        recent_gf_3[away].append(ftag)
        recent_gf_5[home].append(fthg)
        recent_gf_5[away].append(ftag)

        recent_ga_3[home].append(ftag)
        recent_ga_3[away].append(fthg)
        recent_ga_5[home].append(ftag)
        recent_ga_5[away].append(fthg)

        recent_gd_3[home].append(fthg - ftag)
        recent_gd_3[away].append(ftag - fthg)
        recent_gd_5[home].append(fthg - ftag)
        recent_gd_5[away].append(ftag - fthg)

        recent_shots_3[home].append(hs)
        recent_shots_3[away].append(a_s)
        recent_shots_5[home].append(hs)
        recent_shots_5[away].append(a_s)

        recent_sot_3[home].append(hst)
        recent_sot_3[away].append(ast)
        recent_sot_5[home].append(hst)
        recent_sot_5[away].append(ast)

        recent_home_points_3[home].append(home_pts)
        recent_home_points_5[home].append(home_pts)
        recent_away_points_3[away].append(away_pts)
        recent_away_points_5[away].append(away_pts)

        last_match_date[home] = date
        last_match_date[away] = date

        h_elo = elo_rating[home]
        a_elo = elo_rating[away]
        h_exp = elo_expected(h_elo, a_elo, home_adv=HOME_ADV)
        a_exp = 1 - h_exp

        h_actual = elo_actual_score(ftr, "home")
        a_actual = elo_actual_score(ftr, "away")

        elo_rating[home] += K * (h_actual - h_exp)
        elo_rating[away] += K * (a_actual - a_exp)

        hh_elo = home_elo_rating[home]
        aa_elo = away_elo_rating[away]
        hh_exp = elo_expected(hh_elo, aa_elo, home_adv=HOME_ADV)
        aa_exp = 1 - hh_exp

        home_elo_rating[home] += K * (h_actual - hh_exp)
        away_elo_rating[away] += K * (a_actual - aa_exp)

    return {
        "overall_stats": overall_stats,
        "home_stats": home_stats,
        "away_stats": away_stats,
        "season_stats": season_stats,
        "recent_points_3": recent_points_3,
        "recent_points_5": recent_points_5,
        "recent_gf_3": recent_gf_3,
        "recent_gf_5": recent_gf_5,
        "recent_ga_3": recent_ga_3,
        "recent_ga_5": recent_ga_5,
        "recent_gd_3": recent_gd_3,
        "recent_gd_5": recent_gd_5,
        "recent_shots_3": recent_shots_3,
        "recent_shots_5": recent_shots_5,
        "recent_sot_3": recent_sot_3,
        "recent_sot_5": recent_sot_5,
        "recent_home_points_3": recent_home_points_3,
        "recent_home_points_5": recent_home_points_5,
        "recent_away_points_3": recent_away_points_3,
        "recent_away_points_5": recent_away_points_5,
        "last_match_date": last_match_date,
        "elo_rating": elo_rating,
        "home_elo_rating": home_elo_rating,
        "away_elo_rating": away_elo_rating
    }


full_state = rebuild_full_state_for_prediction(matches)


# ---------------------------------------------------------------------------
# 18) FIXTURE PREDICTION FUNCTION
# ---------------------------------------------------------------------------
def predict_fixture(home_team, away_team, pipeline_to_use, pipeline_name):
    """
    Predicts the result probabilities for a future fixture using the same
    feature engineering logic as the training data.
    """

    overall_stats = full_state["overall_stats"]
    home_stats = full_state["home_stats"]
    away_stats = full_state["away_stats"]
    season_stats = full_state["season_stats"]

    recent_points_3 = full_state["recent_points_3"]
    recent_points_5 = full_state["recent_points_5"]
    recent_gf_3 = full_state["recent_gf_3"]
    recent_gf_5 = full_state["recent_gf_5"]
    recent_ga_3 = full_state["recent_ga_3"]
    recent_ga_5 = full_state["recent_ga_5"]
    recent_gd_3 = full_state["recent_gd_3"]
    recent_gd_5 = full_state["recent_gd_5"]
    recent_shots_3 = full_state["recent_shots_3"]
    recent_shots_5 = full_state["recent_shots_5"]
    recent_sot_3 = full_state["recent_sot_3"]
    recent_sot_5 = full_state["recent_sot_5"]

    recent_home_points_3 = full_state["recent_home_points_3"]
    recent_home_points_5 = full_state["recent_home_points_5"]
    recent_away_points_3 = full_state["recent_away_points_3"]
    recent_away_points_5 = full_state["recent_away_points_5"]

    last_match_date = full_state["last_match_date"]

    elo_rating = full_state["elo_rating"]
    home_elo_rating = full_state["home_elo_rating"]
    away_elo_rating = full_state["away_elo_rating"]

    current_season = matches["SourceFile"].iloc[-1]
    current_date = matches["Date"].max() + pd.Timedelta(days=7)

    h_all = overall_stats[home_team]
    a_all = overall_stats[away_team]
    h_home = home_stats[home_team]
    a_away = away_stats[away_team]
    h_season = season_stats[current_season][home_team]
    a_season = season_stats[current_season][away_team]

    # Current table positions
    season_table = []
    for team_name, stats in season_stats[current_season].items():
        points = stats["points"]
        gd = stats["gf"] - stats["ga"]
        gf = stats["gf"]
        season_table.append((team_name, points, gd, gf))

    season_table_sorted = sorted(
        season_table,
        key=lambda x: (x[1], x[2], x[3], x[0]),
        reverse=True
    )

    table_position_lookup = {}
    for pos, (team_name, _, _, _) in enumerate(season_table_sorted, start=1):
        table_position_lookup[team_name] = pos

    home_table_pos = table_position_lookup.get(home_team, 20)
    away_table_pos = table_position_lookup.get(away_team, 20)

    home_rest_days = days_since(last_match_date[home_team], current_date, default=7.0)
    away_rest_days = days_since(last_match_date[away_team], current_date, default=7.0)

    HOME_ADV = 60
    home_elo = elo_rating[home_team]
    away_elo = elo_rating[away_team]
    home_home_elo = home_elo_rating[home_team]
    away_away_elo = away_elo_rating[away_team]

    home_elo_expected = elo_expected(home_elo, away_elo, home_adv=HOME_ADV)
    away_elo_expected = 1 - home_elo_expected
    home_venue_elo_expected = elo_expected(home_home_elo, away_away_elo, home_adv=HOME_ADV)
    away_venue_elo_expected = 1 - home_venue_elo_expected

    row = {
        "HomeTeam": home_team,
        "AwayTeam": away_team,

        "home_ppg_all": safe_avg(h_all["points"], h_all["played"], 1.35),
        "away_ppg_all": safe_avg(a_all["points"], a_all["played"], 1.10),

        "home_win_rate_all": safe_avg(h_all["wins"], h_all["played"], 0.35),
        "away_win_rate_all": safe_avg(a_all["wins"], a_all["played"], 0.30),

        "home_draw_rate_all": safe_avg(h_all["draws"], h_all["played"], 0.25),
        "away_draw_rate_all": safe_avg(a_all["draws"], a_all["played"], 0.25),

        "home_avg_gf_all": safe_avg(h_all["gf"], h_all["played"], 1.40),
        "away_avg_gf_all": safe_avg(a_all["gf"], a_all["played"], 1.10),

        "home_avg_ga_all": safe_avg(h_all["ga"], h_all["played"], 1.20),
        "away_avg_ga_all": safe_avg(a_all["ga"], a_all["played"], 1.30),

        "home_avg_shots_all": safe_avg(h_all["shots"], h_all["played"], 12.0),
        "away_avg_shots_all": safe_avg(a_all["shots"], a_all["played"], 11.0),

        "home_avg_sot_all": safe_avg(h_all["sot"], h_all["played"], 4.5),
        "away_avg_sot_all": safe_avg(a_all["sot"], a_all["played"], 4.0),

        "home_ppg_home": safe_avg(h_home["points"], h_home["played"], 1.50),
        "away_ppg_away": safe_avg(a_away["points"], a_away["played"], 0.95),

        "home_win_rate_home": safe_avg(h_home["wins"], h_home["played"], 0.45),
        "away_win_rate_away": safe_avg(a_away["wins"], a_away["played"], 0.25),

        "home_avg_gf_home": safe_avg(h_home["gf"], h_home["played"], 1.55),
        "away_avg_gf_away": safe_avg(a_away["gf"], a_away["played"], 1.00),

        "home_avg_ga_home": safe_avg(h_home["ga"], h_home["played"], 1.10),
        "away_avg_ga_away": safe_avg(a_away["ga"], a_away["played"], 1.45),

        "home_avg_shots_home": safe_avg(h_home["shots"], h_home["played"], 13.0),
        "away_avg_shots_away": safe_avg(a_away["shots"], a_away["played"], 10.5),

        "home_avg_sot_home": safe_avg(h_home["sot"], h_home["played"], 4.8),
        "away_avg_sot_away": safe_avg(a_away["sot"], a_away["played"], 3.8),

        "home_season_ppg": safe_avg(h_season["points"], h_season["played"], 1.35),
        "away_season_ppg": safe_avg(a_season["points"], a_season["played"], 1.10),

        "home_season_gf_pg": safe_avg(h_season["gf"], h_season["played"], 1.40),
        "away_season_gf_pg": safe_avg(a_season["gf"], a_season["played"], 1.10),

        "home_season_ga_pg": safe_avg(h_season["ga"], h_season["played"], 1.20),
        "away_season_ga_pg": safe_avg(a_season["ga"], a_season["played"], 1.30),

        "home_season_gd_pg": safe_avg(h_season["gf"] - h_season["ga"], h_season["played"], 0.10),
        "away_season_gd_pg": safe_avg(a_season["gf"] - a_season["ga"], a_season["played"], -0.10),

        "home_season_shots_pg": safe_avg(h_season["shots"], h_season["played"], 12.0),
        "away_season_shots_pg": safe_avg(a_season["shots"], a_season["played"], 11.0),

        "home_season_sot_pg": safe_avg(h_season["sot"], h_season["played"], 4.5),
        "away_season_sot_pg": safe_avg(a_season["sot"], a_season["played"], 4.0),

        "home_table_pos": home_table_pos,
        "away_table_pos": away_table_pos,

        "home_form_pts_3": deque_mean(recent_points_3[home_team], 1.30),
        "away_form_pts_3": deque_mean(recent_points_3[away_team], 1.10),

        "home_form_gf_3": deque_mean(recent_gf_3[home_team], 1.40),
        "away_form_gf_3": deque_mean(recent_gf_3[away_team], 1.10),

        "home_form_ga_3": deque_mean(recent_ga_3[home_team], 1.20),
        "away_form_ga_3": deque_mean(recent_ga_3[away_team], 1.30),

        "home_form_gd_3": deque_mean(recent_gd_3[home_team], 0.10),
        "away_form_gd_3": deque_mean(recent_gd_3[away_team], -0.10),

        "home_form_shots_3": deque_mean(recent_shots_3[home_team], 12.0),
        "away_form_shots_3": deque_mean(recent_shots_3[away_team], 11.0),

        "home_form_sot_3": deque_mean(recent_sot_3[home_team], 4.5),
        "away_form_sot_3": deque_mean(recent_sot_3[away_team], 4.0),

        "home_form_pts_5": deque_mean(recent_points_5[home_team], 1.30),
        "away_form_pts_5": deque_mean(recent_points_5[away_team], 1.10),

        "home_form_gf_5": deque_mean(recent_gf_5[home_team], 1.40),
        "away_form_gf_5": deque_mean(recent_gf_5[away_team], 1.10),

        "home_form_ga_5": deque_mean(recent_ga_5[home_team], 1.20),
        "away_form_ga_5": deque_mean(recent_ga_5[away_team], 1.30),

        "home_form_gd_5": deque_mean(recent_gd_5[home_team], 0.10),
        "away_form_gd_5": deque_mean(recent_gd_5[away_team], -0.10),

        "home_form_shots_5": deque_mean(recent_shots_5[home_team], 12.0),
        "away_form_shots_5": deque_mean(recent_shots_5[away_team], 11.0),

        "home_form_sot_5": deque_mean(recent_sot_5[home_team], 4.5),
        "away_form_sot_5": deque_mean(recent_sot_5[away_team], 4.0),

        "home_home_form_pts_3": deque_mean(recent_home_points_3[home_team], 1.45),
        "home_home_form_pts_5": deque_mean(recent_home_points_5[home_team], 1.45),

        "away_away_form_pts_3": deque_mean(recent_away_points_3[away_team], 1.00),
        "away_away_form_pts_5": deque_mean(recent_away_points_5[away_team], 1.00),

        "home_elo": home_elo,
        "away_elo": away_elo,
        "home_home_elo": home_home_elo,
        "away_away_elo": away_away_elo,
        "home_elo_expected": home_elo_expected,
        "away_elo_expected": away_elo_expected,
        "home_venue_elo_expected": home_venue_elo_expected,
        "away_venue_elo_expected": away_venue_elo_expected,

        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
    }

    diff_pairs = [
        ("ppg_all_diff", "home_ppg_all", "away_ppg_all"),
        ("win_rate_all_diff", "home_win_rate_all", "away_win_rate_all"),
        ("draw_rate_all_diff", "home_draw_rate_all", "away_draw_rate_all"),
        ("avg_gf_all_diff", "home_avg_gf_all", "away_avg_gf_all"),
        ("avg_ga_all_diff", "home_avg_ga_all", "away_avg_ga_all"),
        ("avg_shots_all_diff", "home_avg_shots_all", "away_avg_shots_all"),
        ("avg_sot_all_diff", "home_avg_sot_all", "away_avg_sot_all"),

        ("ppg_venue_diff", "home_ppg_home", "away_ppg_away"),
        ("win_rate_venue_diff", "home_win_rate_home", "away_win_rate_away"),
        ("gf_venue_diff", "home_avg_gf_home", "away_avg_gf_away"),
        ("ga_venue_diff", "home_avg_ga_home", "away_avg_ga_away"),
        ("shots_venue_diff", "home_avg_shots_home", "away_avg_shots_away"),
        ("sot_venue_diff", "home_avg_sot_home", "away_avg_sot_away"),

        ("season_ppg_diff", "home_season_ppg", "away_season_ppg"),
        ("season_gf_pg_diff", "home_season_gf_pg", "away_season_gf_pg"),
        ("season_ga_pg_diff", "home_season_ga_pg", "away_season_ga_pg"),
        ("season_gd_pg_diff", "home_season_gd_pg", "away_season_gd_pg"),
        ("season_shots_pg_diff", "home_season_shots_pg", "away_season_shots_pg"),
        ("season_sot_pg_diff", "home_season_sot_pg", "away_season_sot_pg"),

        ("form_pts_3_diff", "home_form_pts_3", "away_form_pts_3"),
        ("form_pts_5_diff", "home_form_pts_5", "away_form_pts_5"),

        ("form_gf_3_diff", "home_form_gf_3", "away_form_gf_3"),
        ("form_gf_5_diff", "home_form_gf_5", "away_form_gf_5"),

        ("form_ga_3_diff", "home_form_ga_3", "away_form_ga_3"),
        ("form_ga_5_diff", "home_form_ga_5", "away_form_ga_5"),

        ("form_gd_3_diff", "home_form_gd_3", "away_form_gd_3"),
        ("form_gd_5_diff", "home_form_gd_5", "away_form_gd_5"),

        ("form_shots_3_diff", "home_form_shots_3", "away_form_shots_3"),
        ("form_shots_5_diff", "home_form_shots_5", "away_form_shots_5"),

        ("form_sot_3_diff", "home_form_sot_3", "away_form_sot_3"),
        ("form_sot_5_diff", "home_form_sot_5", "away_form_sot_5"),

        ("home_away_form_pts_3_diff", "home_home_form_pts_3", "away_away_form_pts_3"),
        ("home_away_form_pts_5_diff", "home_home_form_pts_5", "away_away_form_pts_5"),

        ("elo_diff", "home_elo", "away_elo"),
        ("venue_elo_diff", "home_home_elo", "away_away_elo"),
        ("elo_expected_diff", "home_elo_expected", "away_elo_expected"),
        ("venue_elo_expected_diff", "home_venue_elo_expected", "away_venue_elo_expected"),

        ("rest_days_diff", "home_rest_days", "away_rest_days"),
    ]

    for diff_name, left, right in diff_pairs:
        row[diff_name] = row[left] - row[right]

    row["table_pos_diff"] = row["away_table_pos"] - row["home_table_pos"]

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


# ---------------------------------------------------------------------------
# 19) EXAMPLE FUTURE FIXTURE PREDICTIONS
# ---------------------------------------------------------------------------
predict_fixture("Arsenal", "Chelsea", best_pipeline, best_model_name)
predict_fixture("Liverpool", "Man City", best_pipeline, best_model_name)