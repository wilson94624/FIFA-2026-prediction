import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import sys
sys.path.append("/Users/wilson/Desktop/FIFA/src")
from data_preprocessing import load_data, preprocess_historical_matches

MODEL_DIR = "/Users/wilson/Desktop/FIFA/src"

def train_and_evaluate():
    matches, appearances, features_2026, _ = load_data()
    X, y = preprocess_historical_matches(matches, appearances)
    
    # 定義模型
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'BoostedTrees': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # 進行 10-fold 交叉驗證
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    cv_results = {name: [] for name in models.keys()}
    
    print("=========================================")
    print("10-Fold Cross-Validation Results:")
    print("=========================================")
    
    for name, model in models.items():
        accuracies = []
        f1s = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 訓練
            model.fit(X_train, y_train)
            # 預測
            y_pred = model.predict(X_test)
            
            accuracies.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred, average='weighted'))
            
        cv_results[name] = accuracies
        print(f"Model: {name}")
        print(f"  Mean Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
        print(f"  Mean F1-Score: {np.mean(f1s):.4f}")
        print("-----------------------------------------")
        
    # 保存交叉驗證的 accuracy 序列，供後續 t-test 使用
    with open(os.path.join(MODEL_DIR, "cv_accuracies.pkl"), "wb") as f:
        pickle.dump(cv_results, f)
        
    # 在全資料集上重新訓練並保存模型
    print("\nTraining on full historical dataset and saving models...")
    for name, model in models.items():
        model.fit(X, y)
        model_path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved {name} model to {model_path}")
        
        # 評估在訓練集上的 confusion matrix
        y_pred_full = model.predict(X)
        print(f"\nClassification Report for {name} on Full Dataset:")
        print(classification_report(y, y_pred_full, target_names=['Draw', 'Home Win', 'Away Win']))
        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred_full))
        print("=========================================")

if __name__ == '__main__':
    train_and_evaluate()
