import os
import pandas as pd
import numpy as np

# 設定路徑
DATA_DIR = "/Users/wilson/Desktop/FIFA/archive"

def to_int(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip().lower()
    if val_str in ['no', 'none', 'na', 'nan', 'null', '']:
        return 0
    try:
        return int(float(val_str))
    except:
        return 0

def load_data():
    matches = pd.read_csv(os.path.join(DATA_DIR, "wc_matches_historical.csv"))
    appearances = pd.read_csv(os.path.join(DATA_DIR, "wc_team_appearances.csv"))
    features_2026 = pd.read_csv(os.path.join(DATA_DIR, "wc_prediction_features_2026.csv"))
    groups_2026 = pd.read_csv(os.path.join(DATA_DIR, "wc_2026_groups.csv"))
    return matches, appearances, features_2026, groups_2026

# ==========================================
# 第一版 (Baseline) 專用函數 (保持原封不動)
# ==========================================
def preprocess_historical_matches(matches, appearances):
    lookup = {}
    for _, row in appearances.iterrows():
        team_name = str(row['team']).strip()
        lookup[(team_name, int(row['wc_year']))] = {
            'titles': to_int(row['wc_titles_before_tournament']),
            'experience': to_int(row['consecutive_appearances']),
            'is_host': 1 if str(row['host_nation']).strip().lower() in ['yes', '1', 'true'] else 0
        }
        
    X_list = []
    y_list = []
    
    for idx, row in matches.iterrows():
        year = int(row['wc_year'])
        home = str(row['home_team']).strip()
        away = str(row['away_team']).strip()
        
        home_feats = lookup.get((home, year), {'titles': 0, 'experience': 0, 'is_host': 0})
        away_feats = lookup.get((away, year), {'titles': 0, 'experience': 0, 'is_host': 0})
        
        home_elo = row['home_pre_match_elo']
        away_elo = row['away_pre_match_elo']
        
        if pd.isna(home_elo): home_elo = 1500.0
        if pd.isna(away_elo): away_elo = 1500.0
            
        elo_diff = home_elo - away_elo
        titles_diff = home_feats['titles'] - away_feats['titles']
        experience_diff = home_feats['experience'] - away_feats['experience']
        
        home_is_host = home_feats['is_host']
        away_is_host = away_feats['is_host']
        
        res = row['result_type']
        if res == 'Home Win':
            label = 1
        elif res == 'Away Win':
            label = 2
        else:
            label = 0
            
        X_list.append({
            'elo_diff': elo_diff,
            'titles_diff': titles_diff,
            'experience_diff': experience_diff,
            'home_is_host': home_is_host,
            'away_is_host': away_is_host
        })
        y_list.append(label)
        
    df_X = pd.DataFrame(X_list)
    df_y = pd.Series(y_list, name='label')
    return df_X, df_y

def get_team_features_2026(features_2026):
    team_features = {}
    for _, row in features_2026.iterrows():
        team_name = str(row['team']).strip()
        team_features[team_name] = {
            'elo': float(row['elo_rating_2026']) if not pd.isna(row['elo_rating_2026']) else 1500.0,
            'titles': to_int(row['titles_before_2026']),
            'experience': to_int(row['consecutive_wc_appearances']),
            'is_host': 1 if str(row['host_nation']).strip().lower() in ['yes', '1', 'true'] else 0
        }
    return team_features

def build_match_features_2026(team_a, team_b, team_features_2026):
    feat_a = team_features_2026.get(team_a, {'elo': 1500.0, 'titles': 0, 'experience': 0, 'is_host': 0})
    feat_b = team_features_2026.get(team_b, {'elo': 1500.0, 'titles': 0, 'experience': 0, 'is_host': 0})
    
    elo_diff = feat_a['elo'] - feat_b['elo']
    titles_diff = feat_a['titles'] - feat_b['titles']
    experience_diff = feat_a['experience'] - feat_b['experience']
    
    home_is_host = feat_a['is_host']
    away_is_host = feat_b['is_host']
    
    return pd.DataFrame([{
        'elo_diff': elo_diff,
        'titles_diff': titles_diff,
        'experience_diff': experience_diff,
        'home_is_host': home_is_host,
        'away_is_host': away_is_host
    }])

# ==========================================
# 第二版 (Poisson 迴歸) 專用函數 (NEW)
# ==========================================
def preprocess_historical_matches_poisson(matches, appearances):
    # 建立查找字典
    lookup = {}
    for _, row in appearances.iterrows():
        team_name = str(row['team']).strip()
        played = row['matches_played'] if not pd.isna(row['matches_played']) and row['matches_played'] > 0 else 1.0
        gs = row['goals_scored'] if not pd.isna(row['goals_scored']) else 0.0
        ga = row['goals_conceded'] if not pd.isna(row['goals_conceded']) else 0.0
        
        lookup[(team_name, int(row['wc_year']))] = {
            'titles': to_int(row['wc_titles_before_tournament']),
            'experience': to_int(row['consecutive_appearances']),
            'is_host': 1 if str(row['host_nation']).strip().lower() in ['yes', '1', 'true'] else 0,
            'attack_strength': gs / played,  # 當屆場均進球
            'defense_strength': ga / played  # 當屆場均失球
        }
        
    X_list = []
    y_home_list = []
    y_away_list = []
    
    for idx, row in matches.iterrows():
        year = int(row['wc_year'])
        home = str(row['home_team']).strip()
        away = str(row['away_team']).strip()
        
        # 預設場均值
        default_stats = {'titles': 0, 'experience': 0, 'is_host': 0, 'attack_strength': 1.2, 'defense_strength': 1.2}
        home_feats = lookup.get((home, year), default_stats)
        away_feats = lookup.get((away, year), default_stats)
        
        home_elo = row['home_pre_match_elo']
        away_elo = row['away_pre_match_elo']
        
        if pd.isna(home_elo): home_elo = 1500.0
        if pd.isna(away_elo): away_elo = 1500.0
            
        elo_diff = (home_elo - away_elo) / 100.0
        titles_diff = home_feats['titles'] - away_feats['titles']
        experience_diff = home_feats['experience'] - away_feats['experience']
        
        home_is_host = home_feats['is_host']
        away_is_host = away_feats['is_host']
        
        # 進攻與防守強度特徵
        home_att = home_feats['attack_strength']
        home_def = home_feats['defense_strength']
        away_att = away_feats['attack_strength']
        away_def = away_feats['defense_strength']
        
        # 實際進球數 (Label)
        home_goals = int(row['home_goals']) if not pd.isna(row['home_goals']) else 0
        away_goals = int(row['away_goals']) if not pd.isna(row['away_goals']) else 0
        
        X_list.append({
            'elo_diff': elo_diff,
            'titles_diff': titles_diff,
            'experience_diff': experience_diff,
            'home_is_host': home_is_host,
            'away_is_host': away_is_host,
            'home_attack': home_att,
            'home_defense': home_def,
            'away_attack': away_att,
            'away_defense': away_def
        })
        y_home_list.append(home_goals)
        y_away_list.append(away_goals)
        
    df_X = pd.DataFrame(X_list)
    df_y_home = pd.Series(y_home_list, name='home_goals')
    df_y_away = pd.Series(y_away_list, name='away_goals')
    return df_X, df_y_home, df_y_away

def get_team_features_2026_poisson(features_2026, qualifying_summary):
    # 建立外圍賽查找字典
    qual_lookup = {}
    for _, row in qualifying_summary.iterrows():
        team_name = str(row['team']).strip()
        played = float(row['qualifying_played']) if not pd.isna(row['qualifying_played']) and row['qualifying_played'] > 0 else 1.0
        qual_lookup[team_name] = {
            'played': played,
            'gf': float(row['qualifying_gf']) if not pd.isna(row['qualifying_gf']) else 12.0,
            'ga': float(row['qualifying_ga']) if not pd.isna(row['qualifying_ga']) else 12.0
        }
        
    # 計算全體外圍賽的平均場均進球與失球，做為東道主的預設值
    all_atts = []
    all_defs = []
    for team, data in qual_lookup.items():
        all_atts.append(data['gf'] / data['played'])
        all_defs.append(data['ga'] / data['played'])
    global_avg_att = np.mean(all_atts) if all_atts else 1.2
    global_avg_def = np.mean(all_defs) if all_defs else 1.2
    
    team_features = {}
    for _, row in features_2026.iterrows():
        team_name = str(row['team']).strip()
        
        # 取得外圍賽場均數據
        qual_data = qual_lookup.get(team_name, {'played': 1.0, 'gf': global_avg_att, 'ga': global_avg_def})
        played = qual_data['played']
        att_str = qual_data['gf'] / played
        def_str = qual_data['ga'] / played
        
        # 美、加、墨身為東道主，外圍賽數據若缺失，直接給全體平均
        is_host_val = 1 if str(row['host_nation']).strip().lower() in ['yes', '1', 'true'] else 0
        if is_host_val == 1:
            att_str = global_avg_att
            def_str = global_avg_def
            
        team_features[team_name] = {
            'elo': float(row['elo_rating_2026']) if not pd.isna(row['elo_rating_2026']) else 1500.0,
            'titles': to_int(row['titles_before_2026']),
            'experience': to_int(row['consecutive_wc_appearances']),
            'is_host': is_host_val,
            'attack_strength': att_str,
            'defense_strength': def_str
        }
    return team_features

def build_match_features_2026_poisson(team_a, team_b, team_features_2026):
    feat_a = team_features_2026.get(team_a, {'elo': 1500.0, 'titles': 0, 'experience': 0, 'is_host': 0, 'attack_strength': 1.2, 'defense_strength': 1.2})
    feat_b = team_features_2026.get(team_b, {'elo': 1500.0, 'titles': 0, 'experience': 0, 'is_host': 0, 'attack_strength': 1.2, 'defense_strength': 1.2})
    
    elo_diff = (feat_a['elo'] - feat_b['elo']) / 100.0
    titles_diff = feat_a['titles'] - feat_b['titles']
    experience_diff = feat_a['experience'] - feat_b['experience']
    
    home_is_host = feat_a['is_host']
    away_is_host = feat_b['is_host']
    
    home_att = feat_a['attack_strength']
    home_def = feat_a['defense_strength']
    away_att = feat_b['attack_strength']
    away_def = feat_b['defense_strength']
    
    return pd.DataFrame([{
        'elo_diff': elo_diff,
        'titles_diff': titles_diff,
        'experience_diff': experience_diff,
        'home_is_host': home_is_host,
        'away_is_host': away_is_host,
        'home_attack': home_att,
        'home_defense': home_def,
        'away_attack': away_att,
        'away_defense': away_def
    }])

if __name__ == '__main__':
    # 測試
    matches, appearances, features_2026, groups_2026 = load_data()
    qual_summary = pd.read_csv(os.path.join(DATA_DIR, "wc_2026_qualifying_summary.csv"))
    
    X_pois, y_h, y_a = preprocess_historical_matches_poisson(matches, appearances)
    team_feats_pois = get_team_features_2026_poisson(features_2026, qual_summary)
    
    print("Poisson features shape:", X_pois.shape)
    print("Poisson sample data:")
    print(X_pois.head(2))
    print("Poisson sample labels (home_goals / away_goals):")
    print(f"Home goals mean: {y_h.mean():.2f}, Away goals mean: {y_a.mean():.2f}")
