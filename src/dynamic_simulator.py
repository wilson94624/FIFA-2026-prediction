import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import poisson

import sys
sys.path.append("/Users/wilson/Desktop/FIFA/src")
from data_preprocessing import load_data, get_team_features_2026_poisson

MODEL_DIR = "/Users/wilson/Desktop/FIFA/src"
DATA_DIR = "/Users/wilson/Desktop/FIFA/archive"
K_FACTOR = 60.0  # 世界盃決賽圈的 Elo K-factor

def get_elo_expected_score(elo_a, elo_b):
    # 計算 A 對 B 的預期得分 (E_A)
    return 1.0 / (10.0 ** ((elo_b - elo_a) / 400.0) + 1.0)

def update_elo(elo_a, elo_b, actual_a, actual_b):
    # actual_a: 贏=1.0, 平=0.5, 輸=0.0
    e_a = get_elo_expected_score(elo_a, elo_b)
    e_b = 1.0 - e_a
    
    new_elo_a = elo_a + K_FACTOR * (actual_a - e_a)
    new_elo_b = elo_b + K_FACTOR * (actual_b - e_b)
    return new_elo_a, new_elo_b

def simulate_match_poisson(team_a, team_b, team_features, coef_h, intercept_h, coef_a, intercept_a, scaler_mean, scaler_scale, rho, is_knockout=False):
    feat_a = team_features[team_a]
    feat_b = team_features[team_b]
    
    # 建立特徵 (9維)
    # 這裡的 elo_diff 在前處理中已被除以 100.0
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
    
    # 使用 NumPy 進行特徵標準化與預測，避開 sklearn 的大開銷
    x_scaled = (x - scaler_mean) / scaler_scale
    l = np.exp(np.dot(coef_h, x_scaled) + intercept_h)
    m = np.exp(np.dot(coef_a, x_scaled) + intercept_a)
    
    l = max(l, 0.05)
    m = max(m, 0.05)
    
    # 建立 6x6 比分機率矩陣
    # 為了提速，我們只計算至多 5 球 (絕大多數比賽進球數在 0-5 之間)
    max_goals = 5
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    
    # 用純數學計算代替 scipy.stats.poisson.pmf，速度快上千倍
    exp_neg_l = np.exp(-l)
    p_x = [
        exp_neg_l,
        l * exp_neg_l,
        (l ** 2) / 2.0 * exp_neg_l,
        (l ** 3) / 6.0 * exp_neg_l,
        (l ** 4) / 24.0 * exp_neg_l,
        (l ** 5) / 120.0 * exp_neg_l
    ]
    
    exp_neg_m = np.exp(-m)
    p_y = [
        exp_neg_m,
        m * exp_neg_m,
        (m ** 2) / 2.0 * exp_neg_m,
        (m ** 3) / 6.0 * exp_neg_m,
        (m ** 4) / 24.0 * exp_neg_m,
        (m ** 5) / 120.0 * exp_neg_m
    ]
    
    for x_g in range(max_goals + 1):
        for y_g in range(max_goals + 1):
            px = p_x[x_g]
            py = p_y[y_g]
            
            # Dixon-Coles 修正
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
            
    # 歸一化
    total = np.sum(matrix)
    if total > 0:
        matrix /= total
    else:
        matrix = np.ones((max_goals + 1, max_goals + 1)) / 36.0
        
    # 扁平化抽樣比分
    flat_probs = matrix.flatten()
    idx = np.random.choice(36, p=flat_probs)
    goals_a = idx // 6
    goals_b = idx % 6
    
    # 根據比分判定常規時間結果
    if goals_a > goals_b:
        actual_a, actual_b = 1.0, 0.0
        winner = team_a
    elif goals_b > goals_a:
        actual_a, actual_b = 0.0, 1.0
        winner = team_b
    else:
        actual_a, actual_b = 0.5, 0.5
        winner = None
        
    # 動態更新 Elo (只根據常規時間結果計算)
    new_elo_a, new_elo_b = update_elo(feat_a['elo'], feat_b['elo'], actual_a, actual_b)
    feat_a['elo'] = new_elo_a
    feat_b['elo'] = new_elo_b
    
    if is_knockout and winner is None:
        # 淘汰賽平手時，進行 PK 大戰
        # 利用 Elo 預期勝率作為 PK 戰獲勝的隨機機率
        p_a_pk = get_elo_expected_score(new_elo_a, new_elo_b)
        winner = np.random.choice([team_a, team_b], p=[p_a_pk, 1.0 - p_a_pk])
        
    return goals_a, goals_b, winner

def simulate_group_stage(groups_df, team_features, coef_h, intercept_h, coef_a, intercept_a, scaler_mean, scaler_scale, rho):
    groups = groups_df['group'].unique()
    
    # 建立每隊小組賽的統計數據
    standings = {}
    for _, row in groups_df.iterrows():
        t_name = str(row['team']).strip()
        standings[t_name] = {
            'group': row['group'],
            'points': 0,
            'gd': 0,  # 淨勝球 Goal Difference
            'gs': 0   # 總進球數 Goals Scored
        }
        
    for grp in groups:
        grp_teams = groups_df[groups_df['group'] == grp]['team'].tolist()
        for i in range(len(grp_teams)):
            for j in range(i + 1, len(grp_teams)):
                team_a = grp_teams[i]
                team_b = grp_teams[j]
                
                # 模擬小組賽 (is_knockout = False)
                ga, gb, win_team = simulate_match_poisson(
                    team_a, team_b, team_features,
                    coef_h, intercept_h, coef_a, intercept_a,
                    scaler_mean, scaler_scale, rho, is_knockout=False
                )
                
                # 更新積分、淨勝球、進球數
                standings[team_a]['gs'] += ga
                standings[team_a]['gd'] += (ga - gb)
                standings[team_b]['gs'] += gb
                standings[team_b]['gd'] += (gb - ga)
                
                if win_team == team_a:
                    standings[team_a]['points'] += 3
                elif win_team == team_b:
                    standings[team_b]['points'] += 3
                else:
                    standings[team_a]['points'] += 1
                    standings[team_b]['points'] += 1
                    
    # 小組賽名次計算
    group_results = {grp: [] for grp in groups}
    for team, stats in standings.items():
        group_results[stats['group']].append((team, stats['points'], stats['gd'], stats['gs'], team_features[team]['elo']))
        
    qualified_teams_1st = []
    qualified_teams_2nd = []
    third_place_teams = []
    
    for grp in groups:
        # 排序：Points -> GD -> GS -> Current ELO
        sorted_teams = sorted(group_results[grp], key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
        qualified_teams_1st.append(sorted_teams[0][0])
        qualified_teams_2nd.append(sorted_teams[1][0])
        third_place_teams.append(sorted_teams[2])
        
    # 成績最好的 8 個小組第三名晉級
    sorted_thirds = sorted(third_place_teams, key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
    qualified_thirds = [x[0] for x in sorted_thirds[:8]]
    
    return qualified_teams_1st, qualified_teams_2nd, qualified_thirds

def simulate_tournament_once(groups_df, team_features_init, coef_h, intercept_h, coef_a, intercept_a, scaler_mean, scaler_scale, rho):
    # 由於模擬中會動態修改 Elo 積分，我們需要對 team_features 進行 deep copy，以防影響下一輪模擬
    team_features = {team: stats.copy() for team, stats in team_features_init.items()}
    
    # 1. 小組賽
    firsts, seconds, thirds = simulate_group_stage(
        groups_df, team_features,
        coef_h, intercept_h, coef_a, intercept_a,
        scaler_mean, scaler_scale, rho
    )
    
    # 2. 32強對陣種子排序 (以當前最新 Elo 為準)
    firsts_sorted = sorted(firsts, key=lambda x: team_features[x]['elo'], reverse=True)
    seconds_sorted = sorted(seconds, key=lambda x: team_features[x]['elo'], reverse=True)
    thirds_sorted = sorted(thirds, key=lambda x: team_features[x]['elo'], reverse=True)
    
    bracket = firsts_sorted + seconds_sorted + thirds_sorted
    
    stages = {
        'R32': bracket.copy(),
        'R16': [],
        'QF': [],
        'SF': [],
        'Final': [],
        'Winner': None
    }
    
    # 3. 淘汰賽模擬 (is_knockout = True)
    # R32 -> R16
    current_round = stages['R32']
    next_round = []
    for i in range(16):
        team_a = current_round[i]
        team_b = current_round[31 - i]
        _, _, winner = simulate_match_poisson(
            team_a, team_b, team_features,
            coef_h, intercept_h, coef_a, intercept_a,
            scaler_mean, scaler_scale, rho, is_knockout=True
        )
        next_round.append(winner)
    stages['R16'] = next_round
    
    # R16 -> QF
    current_round = stages['R16']
    next_round = []
    for i in range(8):
        team_a = current_round[i]
        team_b = current_round[15 - i]
        _, _, winner = simulate_match_poisson(
            team_a, team_b, team_features,
            coef_h, intercept_h, coef_a, intercept_a,
            scaler_mean, scaler_scale, rho, is_knockout=True
        )
        next_round.append(winner)
    stages['QF'] = next_round
    
    # QF -> SF
    current_round = stages['QF']
    next_round = []
    for i in range(4):
        team_a = current_round[i]
        team_b = current_round[7 - i]
        _, _, winner = simulate_match_poisson(
            team_a, team_b, team_features,
            coef_h, intercept_h, coef_a, intercept_a,
            scaler_mean, scaler_scale, rho, is_knockout=True
        )
        next_round.append(winner)
    stages['SF'] = next_round
    
    # SF -> Final
    current_round = stages['SF']
    next_round = []
    for i in range(2):
        team_a = current_round[i]
        team_b = current_round[3 - i]
        _, _, winner = simulate_match_poisson(
            team_a, team_b, team_features,
            coef_h, intercept_h, coef_a, intercept_a,
            scaler_mean, scaler_scale, rho, is_knockout=True
        )
        next_round.append(winner)
    stages['Final'] = next_round
    
    # Final -> Winner
    _, _, winner = simulate_match_poisson(
        stages['Final'][0], stages['Final'][1], team_features,
        coef_h, intercept_h, coef_a, intercept_a,
        scaler_mean, scaler_scale, rho, is_knockout=True
    )
    stages['Winner'] = winner
    
    return stages

def run_dynamic_simulation(n_simulations=10000):
    # 載入資料
    _, _, features_2026, groups_2026 = load_data()
    qual_summary = pd.read_csv(os.path.join(DATA_DIR, "wc_2026_qualifying_summary.csv"))
    
    # 載入 Poisson 模型與 scaler
    model_path = os.path.join(MODEL_DIR, "poisson_model.pkl")
    if not os.path.exists(model_path):
        print("Error: poisson_model.pkl not found. Run poisson_model.py first.")
        return
        
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        
    # 取得 2026 特徵
    team_features = get_team_features_2026_poisson(features_2026, qual_summary)
    
    # 提取模型係數與 scaler 參數以提升運算速度
    home_model = model_data['home_model']
    away_model = model_data['away_model']
    scaler = model_data['scaler']
    
    coef_h = home_model.coef_
    intercept_h = home_model.intercept_
    coef_a = away_model.coef_
    intercept_a = away_model.intercept_
    
    scaler_mean = scaler.mean_
    scaler_scale = scaler.scale_
    rho = model_data['rho']
    
    print(f"Starting Dynamic Poisson Monte Carlo Simulation ({n_simulations} runs)...")
    
    stats = {}
    for team in team_features.keys():
        stats[team] = {
            'R32_pct': 0.0,
            'R16_pct': 0.0,
            'QF_pct': 0.0,
            'SF_pct': 0.0,
            'Final_pct': 0.0,
            'Winner_pct': 0.0
        }
        
    for _ in range(n_simulations):
        run = simulate_tournament_once(
            groups_2026, team_features,
            coef_h, intercept_h, coef_a, intercept_a,
            scaler_mean, scaler_scale, rho
        )
        
        for team in run['R32']:
            stats[team]['R32_pct'] += 1
        for team in run['R16']:
            stats[team]['R16_pct'] += 1
        for team in run['QF']:
            stats[team]['QF_pct'] += 1
        for team in run['SF']:
            stats[team]['SF_pct'] += 1
        for team in run['Final']:
            stats[team]['Final_pct'] += 1
        stats[run['Winner']]['Winner_pct'] += 1
        
    for team in stats.keys():
        for stage in stats[team].keys():
            stats[team][stage] = (stats[team][stage] / n_simulations) * 100
            
    df_stats = pd.DataFrame.from_dict(stats, orient='index')
    df_stats = df_stats.sort_values(by='Winner_pct', ascending=False)
    
    print("\nTop 15 Teams by Win Probability (Dynamic Poisson Model):")
    print(df_stats.head(15).to_string(formatters={
        'R32_pct': '{:,.2f}%'.format,
        'R16_pct': '{:,.2f}%'.format,
        'QF_pct': '{:,.2f}%'.format,
        'SF_pct': '{:,.2f}%'.format,
        'Final_pct': '{:,.2f}%'.format,
        'Winner_pct': '{:,.2f}%'.format
    }))
    
    df_stats.to_csv(os.path.join(MODEL_DIR, "simulation_results_v2.csv"))
    print(f"\nDynamic simulation results saved to {os.path.join(MODEL_DIR, 'simulation_results_v2.csv')}")

if __name__ == '__main__':
    import time
    start_time = time.time()
    run_dynamic_simulation(10000)
    print(f"Simulation completed in {time.time() - start_time:.2f} seconds.")
