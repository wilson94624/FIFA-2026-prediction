import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append("/Users/wilson/Desktop/FIFA/src")
from data_preprocessing import load_data, to_int, preprocess_historical_matches, preprocess_historical_matches_poisson

MODEL_DIR = "/Users/wilson/Desktop/FIFA/src"
K_FACTOR = 60.0

def get_elo_expected_score(elo_a, elo_b):
    return 1.0 / (10.0 ** ((elo_b - elo_a) / 400.0) + 1.0)

def update_elo(elo_a, elo_b, actual_a, actual_b):
    e_a = get_elo_expected_score(elo_a, elo_b)
    e_b = 1.0 - e_a
    new_elo_a = elo_a + K_FACTOR * (actual_a - e_a)
    new_elo_b = elo_b + K_FACTOR * (actual_b - e_b)
    return new_elo_a, new_elo_b

def fit_dixon_coles_rho(y_home, y_away, lambdas, mus):
    best_rho = 0.0
    max_log_lik = -np.inf
    rhos = np.linspace(-0.5, 0.5, 101)
    for rho in rhos:
        log_lik = 0.0
        valid = True
        for x, y, l, m in zip(y_home, y_away, lambdas, mus):
            if x == 0 and y == 0:
                tau = 1 - rho * l * m
            elif x == 1 and y == 0:
                tau = 1 + rho * m
            elif x == 0 and y == 1:
                tau = 1 + rho * l
            elif x == 1 and y == 1:
                tau = 1 - rho
            else:
                tau = 1.0
            if tau <= 0:
                valid = False
                break
            log_lik += np.log(tau)
        if valid and log_lik > max_log_lik:
            max_log_lik = log_lik
            best_rho = rho
    return best_rho

def load_2022_quarter_finalists(appearances):
    # 載入卡達世界盃八強的數據
    df_2022 = appearances[appearances['wc_year'] == 2022]
    teams_data = {}
    
    for _, row in df_2022.iterrows():
        team_name = str(row['team']).strip()
        played = row['matches_played'] if not pd.isna(row['matches_played']) and row['matches_played'] > 0 else 1.0
        gs = row['goals_scored'] if not pd.isna(row['goals_scored']) else 0.0
        ga = row['goals_conceded'] if not pd.isna(row['goals_conceded']) else 0.0
        
        teams_data[team_name] = {
            'elo': float(row['elo_rating_approx']) if not pd.isna(row['elo_rating_approx']) else 1500.0,
            'titles': to_int(row['wc_titles_before_tournament']),
            'experience': to_int(row['consecutive_appearances']),
            'is_host': 1 if str(row['host_nation']).strip().lower() in ['yes', '1', 'true'] else 0,
            'attack_strength': gs / played,
            'defense_strength': ga / played
        }
        
    return teams_data

# ==========================================
# 模擬單場對決
# ==========================================
def simulate_match_baseline(team_a, team_b, team_features, model):
    feat_a = team_features[team_a]
    feat_b = team_features[team_b]
    
    elo_diff = feat_a['elo'] - feat_b['elo']
    titles_diff = feat_a['titles'] - feat_b['titles']
    experience_diff = feat_a['experience'] - feat_b['experience']
    home_is_host = feat_a['is_host']
    away_is_host = feat_b['is_host']
    
    features = np.array([[elo_diff, titles_diff, experience_diff, home_is_host, away_is_host]])
    probs = model.predict_proba(features)[0]
    
    p_a_win, p_b_win = probs[1], probs[2]
    total_win = p_a_win + p_b_win
    p_a = p_a_win / total_win if total_win > 0 else 0.5
    
    return np.random.choice([team_a, team_b], p=[p_a, 1.0 - p_a])

def simulate_match_poisson(team_a, team_b, team_features, coef_h, intercept_h, coef_a, intercept_a, scaler_mean, scaler_scale, rho):
    feat_a = team_features[team_a]
    feat_b = team_features[team_b]
    
    elo_diff = (feat_a['elo'] - feat_b['elo']) / 100.0
    titles_diff = feat_a['titles'] - feat_b['titles']
    experience_diff = feat_a['experience'] - feat_b['experience']
    home_is_host = feat_a['is_host']
    away_is_host = feat_b['is_host']
    
    home_att = feat_a['attack_strength']
    home_def = feat_a['defense_strength']
    away_att = feat_b['attack_strength']
    away_def = feat_b['defense_strength']
    
    x = np.array([elo_diff, titles_diff, experience_diff, home_is_host, away_is_host, home_att, home_def, away_att, away_def])
    
    x_scaled = (x - scaler_mean) / scaler_scale
    l = np.exp(np.dot(coef_h, x_scaled) + intercept_h)
    m = np.exp(np.dot(coef_a, x_scaled) + intercept_a)
    
    l = max(l, 0.05)
    m = max(m, 0.05)
    
    max_goals = 5
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    
    exp_neg_l = np.exp(-l)
    p_x = [exp_neg_l, l * exp_neg_l, (l ** 2) / 2.0 * exp_neg_l, (l ** 3) / 6.0 * exp_neg_l, (l ** 4) / 24.0 * exp_neg_l, (l ** 5) / 120.0 * exp_neg_l]
    
    exp_neg_m = np.exp(-m)
    p_y = [exp_neg_m, m * exp_neg_m, (m ** 2) / 2.0 * exp_neg_m, (m ** 3) / 6.0 * exp_neg_m, (m ** 4) / 24.0 * exp_neg_m, (m ** 5) / 120.0 * exp_neg_m]
    
    for x_g in range(max_goals + 1):
        for y_g in range(max_goals + 1):
            px = p_x[x_g]
            py = p_y[y_g]
            
            if x_g == 0 and y_g == 0:
                tau = 1 - rho * l * m
            elif x_g == 1 and y_g == 0:
                tau = 1 + rho * m
            elif x_g == 0 and y_g == 1:
                tau = 1 + rho * l
            elif x_g == 1 and y_g == 1:
                tau = 1 - rho
            else:
                tau = 1.0
                
            matrix[x_g, y_g] = max(tau * px * py, 0.0)
            
    total = np.sum(matrix)
    if total > 0:
        matrix /= total
    else:
        matrix = np.ones((max_goals + 1, max_goals + 1)) / 36.0
        
    flat_probs = matrix.flatten()
    idx = np.random.choice(36, p=flat_probs)
    goals_a = idx // 6
    goals_b = idx % 6
    
    if goals_a > goals_b:
        actual_a, actual_b = 1.0, 0.0
        winner = team_a
    elif goals_b > goals_a:
        actual_a, actual_b = 0.0, 1.0
        winner = team_b
    else:
        actual_a, actual_b = 0.5, 0.5
        winner = None
        
    # 動態更新 Elo 評級
    new_elo_a, new_elo_b = update_elo(feat_a['elo'], feat_b['elo'], actual_a, actual_b)
    feat_a['elo'] = new_elo_a
    feat_b['elo'] = new_elo_b
    
    if winner is None:
        p_a_pk = get_elo_expected_score(new_elo_a, new_elo_b)
        winner = np.random.choice([team_a, team_b], p=[p_a_pk, 1.0 - p_a_pk])
        
    return winner

# ==========================================
# 模擬 2022 八強起淘汰賽
# ==========================================
def simulate_2022_knockout(team_features_init, model, coef_h=None, intercept_h=None, coef_a=None, intercept_a=None, scaler_mean=None, scaler_scale=None, rho=None, version='poisson'):
    team_features = {team: stats.copy() for team, stats in team_features_init.items()}
    
    # QF 對決組合
    qf_matchups = [
        ('Croatia', 'Brazil'),       # QF1
        ('Netherlands', 'Argentina'), # QF2
        ('Morocco', 'Portugal'),     # QF3
        ('England', 'France')        # QF4
    ]
    
    # 1. 8 強 (QF) -> 4 強 (SF)
    qf_winners = []
    for team_a, team_b in qf_matchups:
        if version == 'baseline':
            win = simulate_match_baseline(team_a, team_b, team_features, model)
        else:
            win = simulate_match_poisson(team_a, team_b, team_features, coef_h, intercept_h, coef_a, intercept_a, scaler_mean, scaler_scale, rho)
        qf_winners.append(win)
        
    # 2. 4 強 (SF) -> 決賽 (Final)
    sf_matchups = [
        (qf_winners[0], qf_winners[1]), # SF1
        (qf_winners[2], qf_winners[3])  # SF2
    ]
    
    sf_winners = []
    for team_a, team_b in sf_matchups:
        if version == 'baseline':
            win = simulate_match_baseline(team_a, team_b, team_features, model)
        else:
            win = simulate_match_poisson(team_a, team_b, team_features, coef_h, intercept_h, coef_a, intercept_a, scaler_mean, scaler_scale, rho)
        sf_winners.append(win)
        
    # 3. 決賽 -> 冠軍
    if version == 'baseline':
        winner = simulate_match_baseline(sf_winners[0], sf_winners[1], team_features, model)
    else:
        winner = simulate_match_poisson(sf_winners[0], sf_winners[1], team_features, coef_h, intercept_h, coef_a, intercept_a, scaler_mean, scaler_scale, rho)
        
    return {
        'QF': ['Croatia', 'Brazil', 'Netherlands', 'Argentina', 'Morocco', 'Portugal', 'England', 'France'],
        'SF': qf_winners,
        'Final': sf_winners,
        'Winner': winner
    }

# ==========================================
# 執行驗證
# ==========================================
def validate_2022_knockout(n_simulations=10000):
    matches, appearances, _, _ = load_data()
    
    # 1. 篩選 2022 年以前的比賽作為訓練集 (1930 - 2018)
    train_matches = matches[matches['wc_year'] < 2022]
    
    # 訓練第一版 Baseline Model
    print("Training Baseline Model on pre-2022 data...")
    X_base, y_base = preprocess_historical_matches(train_matches, appearances)
    base_model = LogisticRegression(max_iter=1000, random_state=42)
    base_model.fit(X_base, y_base)
    
    # 訓練第二版 Poisson Model (ELO 差值除以 100)
    print("Training Poisson Model on pre-2022 data...")
    train_matches_scaled = train_matches.copy()
    # 歷史 Elo 差標準化由 preprocess_historical_matches_poisson 處理
    X_pois, y_h, y_a = preprocess_historical_matches_poisson(train_matches, appearances)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pois)
    
    home_model = PoissonRegressor(alpha=1.0)
    away_model = PoissonRegressor(alpha=1.0)
    home_model.fit(X_scaled, y_h)
    away_model.fit(X_scaled, y_a)
    
    lambdas = home_model.predict(X_scaled)
    mus = away_model.predict(X_scaled)
    rho = fit_dixon_coles_rho(y_h, y_a, lambdas, mus)
    
    # 載入八強隊伍特徵
    team_features = load_2022_quarter_finalists(appearances)
    
    # 統計指標設定
    target_teams = ['Argentina', 'Morocco', 'France', 'Croatia', 'Brazil', 'Portugal']
    baseline_stats = {t: {'SF_cnt': 0, 'Winner_cnt': 0} for t in target_teams}
    poisson_stats = {t: {'SF_cnt': 0, 'Winner_cnt': 0} for t in target_teams}
    
    print(f"Simulating Qatar 2022 Quarter-Finals onward ({n_simulations} runs)...")
    
    coef_h, intercept_h = home_model.coef_, home_model.intercept_
    coef_a, intercept_a = away_model.coef_, away_model.intercept_
    scaler_mean, scaler_scale = scaler.mean_, scaler.scale_
    
    for _ in range(n_simulations):
        # 1. Baseline
        run_base = simulate_2022_knockout(team_features, base_model, version='baseline')
        for t in target_teams:
            if t in run_base['SF']:
                baseline_stats[t]['SF_cnt'] += 1
            if t == run_base['Winner']:
                baseline_stats[t]['Winner_cnt'] += 1
                
        # 2. Poisson + Dynamic Elo
        run_pois = simulate_2022_knockout(
            team_features, None,
            coef_h, intercept_h, coef_a, intercept_a,
            scaler_mean, scaler_scale, rho, version='poisson'
        )
        for t in target_teams:
            if t in run_pois['SF']:
                poisson_stats[t]['SF_cnt'] += 1
            if t == run_pois['Winner']:
                poisson_stats[t]['Winner_cnt'] += 1
                
    # 整理輸出
    results_list = []
    for t in target_teams:
        results_list.append({
            'Team': t,
            'Baseline_SF_pct': (baseline_stats[t]['SF_cnt'] / n_simulations) * 100,
            'Poisson_SF_pct': (poisson_stats[t]['SF_cnt'] / n_simulations) * 100,
            'Baseline_Winner_pct': (baseline_stats[t]['Winner_cnt'] / n_simulations) * 100,
            'Poisson_Winner_pct': (poisson_stats[t]['Winner_cnt'] / n_simulations) * 100
        })
        
    df_results = pd.DataFrame(results_list)
    print("\n2022 Qatar Knockout Validation Results (Comparison):")
    print(df_results.to_string(index=False, formatters={
        'Baseline_SF_pct': '{:,.2f}%'.format,
        'Poisson_SF_pct': '{:,.2f}%'.format,
        'Baseline_Winner_pct': '{:,.2f}%'.format,
        'Poisson_Winner_pct': '{:,.2f}%'.format
    }))
    
    results_path = os.path.join(MODEL_DIR, "validation_2022_results.csv")
    df_results.to_csv(results_path, index=False)
    print(f"\nValidation results saved to {results_path}")

if __name__ == '__main__':
    validate_2022_knockout(10000) # 執行 10,000 次模擬以獲得穩健的概率
