import os
import pickle
import numpy as np
import pandas as pd

import sys
sys.path.append("/Users/wilson/Desktop/FIFA/src")
from data_preprocessing import load_data, get_team_features_2026

MODEL_DIR = "/Users/wilson/Desktop/FIFA/src"

def simulate_match(team_a, team_b, team_features, coef, intercept, is_knockout=False):
    feat_a = team_features.get(team_a, {'elo': 1500.0, 'titles': 0, 'experience': 0, 'is_host': 0})
    feat_b = team_features.get(team_b, {'elo': 1500.0, 'titles': 0, 'experience': 0, 'is_host': 0})
    
    elo_diff = feat_a['elo'] - feat_b['elo']
    titles_diff = feat_a['titles'] - feat_b['titles']
    experience_diff = feat_a['experience'] - feat_b['experience']
    
    home_is_host = feat_a['is_host']
    away_is_host = feat_b['is_host']
    
    # 特徵向量 (5,)
    x = np.array([elo_diff, titles_diff, experience_diff, home_is_host, away_is_host])
    
    # 純 NumPy 矩陣乘法 z = coef * x + intercept
    z = np.dot(coef, x) + intercept
    
    # Softmax 轉換
    exp_z = np.exp(z - np.max(z))
    probs = exp_z / np.sum(exp_z)
    
    p_draw = probs[0]
    p_a_win = probs[1]
    p_b_win = probs[2]
    
    if is_knockout:
        # 淘汰賽重新歸一化勝率
        total_win_prob = p_a_win + p_b_win
        if total_win_prob == 0:
            p_a = 0.5
            p_b = 0.5
        else:
            p_a = p_a_win / total_win_prob
            p_b = p_b_win / total_win_prob
        
        outcome = np.random.choice([team_a, team_b], p=[p_a, p_b])
        return outcome
    else:
        # 小組賽
        outcome_idx = np.random.choice([0, 1, 2], p=[p_draw, p_a_win, p_b_win])
        if outcome_idx == 1:
            return 'A_WIN'
        elif outcome_idx == 2:
            return 'B_WIN'
        else:
            return 'DRAW'

def simulate_group_stage(groups_df, team_features, coef, intercept):
    groups = groups_df['group'].unique()
    
    standings = {}
    for _, row in groups_df.iterrows():
        team_name = str(row['team']).strip()
        standings[team_name] = {
            'group': row['group'],
            'points': 0,
            'elo': team_features[team_name]['elo']
        }
        
    for grp in groups:
        grp_teams = groups_df[groups_df['group'] == grp]['team'].tolist()
        for i in range(len(grp_teams)):
            for j in range(i + 1, len(grp_teams)):
                team_a = grp_teams[i]
                team_b = grp_teams[j]
                
                res = simulate_match(team_a, team_b, team_features, coef, intercept, is_knockout=False)
                if res == 'A_WIN':
                    standings[team_a]['points'] += 3
                elif res == 'B_WIN':
                    standings[team_b]['points'] += 3
                else:
                    standings[team_a]['points'] += 1
                    standings[team_b]['points'] += 1
                    
    group_results = {grp: [] for grp in groups}
    for team, stats in standings.items():
        group_results[stats['group']].append((team, stats['points'], stats['elo']))
        
    qualified_teams_1st = []
    qualified_teams_2nd = []
    third_place_teams = []
    
    for grp in groups:
        sorted_teams = sorted(group_results[grp], key=lambda x: (x[1], x[2]), reverse=True)
        qualified_teams_1st.append(sorted_teams[0][0])
        qualified_teams_2nd.append(sorted_teams[1][0])
        third_place_teams.append(sorted_teams[2])
        
    sorted_thirds = sorted(third_place_teams, key=lambda x: (x[1], x[2]), reverse=True)
    qualified_thirds = [x[0] for x in sorted_thirds[:8]]
    
    return qualified_teams_1st, qualified_teams_2nd, qualified_thirds

def simulate_tournament_once(groups_df, team_features, coef, intercept):
    firsts, seconds, thirds = simulate_group_stage(groups_df, team_features, coef, intercept)
    
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
    
    # R32 -> R16
    current_round = stages['R32']
    next_round = []
    for i in range(16):
        team_a = current_round[i]
        team_b = current_round[31 - i]
        winner = simulate_match(team_a, team_b, team_features, coef, intercept, is_knockout=True)
        next_round.append(winner)
    stages['R16'] = next_round
    
    # R16 -> QF
    current_round = stages['R16']
    next_round = []
    for i in range(8):
        team_a = current_round[i]
        team_b = current_round[15 - i]
        winner = simulate_match(team_a, team_b, team_features, coef, intercept, is_knockout=True)
        next_round.append(winner)
    stages['QF'] = next_round
    
    # QF -> SF
    current_round = stages['QF']
    next_round = []
    for i in range(4):
        team_a = current_round[i]
        team_b = current_round[7 - i]
        winner = simulate_match(team_a, team_b, team_features, coef, intercept, is_knockout=True)
        next_round.append(winner)
    stages['SF'] = next_round
    
    # SF -> Final
    current_round = stages['SF']
    next_round = []
    for i in range(2):
        team_a = current_round[i]
        team_b = current_round[3 - i]
        winner = simulate_match(team_a, team_b, team_features, coef, intercept, is_knockout=True)
        next_round.append(winner)
    stages['Final'] = next_round
    
    # Final -> Winner
    winner = simulate_match(stages['Final'][0], stages['Final'][1], team_features, coef, intercept, is_knockout=True)
    stages['Winner'] = winner
    
    return stages

def run_simulation(n_simulations=10000):
    _, _, features_2026, groups_2026 = load_data()
    team_features = get_team_features_2026(features_2026)
    
    model_path = os.path.join(MODEL_DIR, "LogisticRegression_model.pkl")
    if not os.path.exists(model_path):
        print("Error: LogisticRegression_model.pkl not found. Run train_models.py first.")
        return
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # 提取特徵係數與截距
    coef = model.coef_
    intercept = model.intercept_
    
    print(f"Starting optimized Monte Carlo Simulation ({n_simulations} runs) using raw NumPy operations...")
    
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
        run = simulate_tournament_once(groups_2026, team_features, coef, intercept)
        
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
    
    print("\nTop 15 Teams by Win Probability:")
    print(df_stats.head(15).to_string(formatters={
        'R32_pct': '{:,.2f}%'.format,
        'R16_pct': '{:,.2f}%'.format,
        'QF_pct': '{:,.2f}%'.format,
        'SF_pct': '{:,.2f}%'.format,
        'Final_pct': '{:,.2f}%'.format,
        'Winner_pct': '{:,.2f}%'.format
    }))
    
    df_stats.to_csv(os.path.join(MODEL_DIR, "simulation_results.csv"))
    print(f"\nSimulation results saved to {os.path.join(MODEL_DIR, 'simulation_results.csv')}")

if __name__ == '__main__':
    import time
    start_time = time.time()
    run_simulation(10000)
    print(f"Simulation completed in {time.time() - start_time:.2f} seconds.")
