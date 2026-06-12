import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson

import sys
sys.path.append("/Users/wilson/Desktop/FIFA/src")
from data_preprocessing import load_data, preprocess_historical_matches_poisson

MODEL_DIR = "/Users/wilson/Desktop/FIFA/src"

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

def train_poisson_models():
    matches, appearances, _, _ = load_data()
    X, y_home, y_away = preprocess_historical_matches_poisson(matches, appearances)
    
    # 數值穩定性優化：使用 StandardScaler 標準化特徵
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 建立與訓練雙卜瓦松迴歸
    home_model = PoissonRegressor(alpha=1.0, max_iter=1000)
    away_model = PoissonRegressor(alpha=1.0, max_iter=1000)
    
    print("Training Double Poisson Regression models with StandardScaler...")
    home_model.fit(X_scaled_df, y_home)
    away_model.fit(X_scaled_df, y_away)
    
    # 預測歷史比分的期望值 lambdas 與 mus
    lambdas = home_model.predict(X_scaled_df)
    mus = away_model.predict(X_scaled_df)
    
    # 擬合 Dixon-Coles rho
    rho = fit_dixon_coles_rho(y_home, y_away, lambdas, mus)
    print(f"Fitted Dixon-Coles correlation coefficient (rho): {rho:.4f}")
    
    # 保存模型與參數，加入 scaler 確保預測時特徵處理一致
    model_data = {
        'home_model': home_model,
        'away_model': away_model,
        'rho': rho,
        'scaler': scaler
    }
    
    model_path = os.path.join(MODEL_DIR, "poisson_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
        
    print(f"Poisson models and parameters saved to {model_path}")
    
    # 評估
    home_mae = np.mean(np.abs(y_home - lambdas))
    away_mae = np.mean(np.abs(y_away - mus))
    print(f"Evaluation:")
    print(f"  Home Goals MAE: {home_mae:.4f}")
    print(f"  Away Goals MAE: {away_mae:.4f}")

# 供外部呼叫的比分機率生成與抽樣函數
def get_score_probs(features_df, model_data, max_goals=5):
    home_model = model_data['home_model']
    away_model = model_data['away_model']
    rho = model_data['rho']
    scaler = model_data['scaler']
    
    # 預測前先標準化特徵
    features_scaled = scaler.transform(features_df)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features_df.columns)
    
    l = home_model.predict(features_scaled_df)[0]
    m = away_model.predict(features_scaled_df)[0]
    
    l = max(l, 0.05)
    m = max(m, 0.05)
    
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            p_x = poisson.pmf(x, l)
            p_y = poisson.pmf(y, m)
            
            # Dixon-Coles 修正
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
                
            matrix[x, y] = max(tau * p_x * p_y, 0.0)
            
    total = np.sum(matrix)
    if total > 0:
        matrix /= total
    else:
        matrix = np.ones((max_goals + 1, max_goals + 1)) / ((max_goals + 1) ** 2)
        
    return matrix

if __name__ == '__main__':
    train_poisson_models()
