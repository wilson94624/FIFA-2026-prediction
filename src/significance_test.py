import os
import pickle
import numpy as np

# 設定
MODEL_DIR = "/Users/wilson/Desktop/FIFA/src"

def paired_t_test(seq_a, seq_b):
    # k = 10
    k = len(seq_a)
    d = np.array(seq_a) - np.array(seq_b)
    d_mean = np.mean(d)
    
    # 計算分母: sqrt( sum((d_i - d_mean)^2) / (k * (k - 1)) )
    sum_sq_diff = np.sum((d - d_mean) ** 2)
    denominator = np.sqrt(sum_sq_diff / (k * (k - 1)))
    
    if denominator == 0:
        t_stat = 0.0
    else:
        t_stat = d_mean / denominator
        
    return t_stat

def run_tests():
    pkl_path = os.path.join(MODEL_DIR, "cv_accuracies.pkl")
    if not os.path.exists(pkl_path):
        print("Error: cv_accuracies.pkl not found. Run train_models.py first.")
        return
        
    with open(pkl_path, "rb") as f:
        cv_results = pickle.load(f)
        
    lr_acc = cv_results['LogisticRegression']
    rf_acc = cv_results['RandomForest']
    bt_acc = cv_results['BoostedTrees']
    
    # t-distribution 臨界值 (df=9, 雙尾 alpha=0.05)
    t_critical = 2.262
    
    comparisons = [
        ('LogisticRegression', 'RandomForest', lr_acc, rf_acc),
        ('LogisticRegression', 'BoostedTrees', lr_acc, bt_acc),
        ('RandomForest', 'BoostedTrees', rf_acc, bt_acc)
    ]
    
    print("=========================================")
    print("Student's Paired t-Test Results (df = 9):")
    print("=========================================")
    print(f"Critical value for alpha = 0.05 (two-tailed): {t_critical}")
    print("-----------------------------------------")
    
    for name_a, name_b, seq_a, seq_b in comparisons:
        t_val = paired_t_test(seq_a, seq_b)
        mean_diff = np.mean(seq_a) - np.mean(seq_b)
        
        is_significant = abs(t_val) > t_critical
        print(f"Comparison: {name_a} vs {name_b}")
        print(f"  Mean Difference: {mean_diff:+.4f}")
        print(f"  Calculated t-value: {t_val:.4f}")
        if is_significant:
            print(f"  Result: Reject Null Hypothesis (Statistically Significant difference!)")
            better_model = name_a if mean_diff > 0 else name_b
            print(f"  Conclusion: {better_model} is significantly better.")
        else:
            print(f"  Result: Fail to Reject Null Hypothesis (Difference is likely due to chance)")
            print(f"  Conclusion: No statistically significant difference.")
        print("-----------------------------------------")

if __name__ == '__main__':
    run_tests()
