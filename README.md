# 🏆 2026 世界盃足球賽預測：從 Baseline 蒙地卡羅到動態卜瓦松模擬器

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本專案為**國立台灣海洋大學機器學習課程期末專題報告**。旨在利用 Kaggle 歷史數據集，透過機器學習與統計物理模擬，預測即將到來的 **2026 年 FIFA 世界盃足球賽** 奪冠機率。

👉 **[點此直接查看成果 Jupyter Notebook](FIFA_2026_Prediction.ipynb)**

---

## 📖 專案背景與故事線

我的預測系統經歷了兩個版本的演進，完整記錄了從「簡單直覺」到「博弈級預測流水線」的優化過程：

### 第一版（Baseline）的反思
最初，我使用傳統的監督式學習三分類器（Logistic Regression, Random Forest）直接預測每場比賽的勝/平/負（W/D/L），並以**靜態 Elo 積分**進行 10,000 次蒙地卡羅賽程模擬。
* **痛點**：模擬結果產生嚴重的「統計複利效應」，強隊（巴西、西班牙）奪冠機率被無限放大（巴西高達 39%），且無法計算具體比分以完成 2026 新制小組第三名路由。這顯然是不合理的「死板」預測。

### 第二版（Optimized Pipeline）的突破
為了克服上述缺陷，我參考了資料科學界與頂級博弈機構（如 Opta, Gracenote）的混合預測架構：
1. **雙卜瓦松迴歸 (Double Poisson Regression)**：改為預測主客兩隊的進球期望值（$\lambda$ 與 $\mu$），引入團隊防守與進攻強度指標。
2. **Dixon-Coles 修正**：使用最大概似估計（MLE）擬合相關性參數 $\rho$，動態調高低比分（0:0, 1:1）下的平手機率。
3. **動態 ELO 狀態更新**：在模擬器的每一場比賽後，即時根據比賽結果更新兩隊的 ELO 評級，完美模擬出「爆冷翻車後的戰力崩盤」或「黑馬連勝後的氣勢如虹」。

---

## 📊 2022 卡達世界盃實戰驗證

我使用 1930 - 2018 年數據訓練模型，在 2022 年卡達世界盃（八強起）進行 10,000 次預測對抗：

| 國家隊 | 2022 實際結果 | Baseline 四強率 | 第二版 (Poisson) 四強率 |
| :---: | :---: | :---: | :---: |
| **阿根廷** | 🏆 **冠軍** | 20.47% | **30.64%** ($\uparrow$ 捕捉低開高走軌跡) |
| **摩洛哥** | 🏅 **四強** | 39.13% | **58.66%** ($\uparrow\uparrow$ 捕捉鋼鐵防守特徵) |
| **巴西** | 八強 (淘汰) | 49.08% | **32.30%** ($\downarrow$ 平滑強隊統計優勢) |

---

## 🛠️ 專案目錄結構

* [FIFA_2026_Prediction.ipynb](FIFA_2026_Prediction.ipynb)：最終成果展示 Jupyter Notebook。
* `src/`：核心演算法與資料預處理目錄。
  * `data_preprocessing.py`：特徵工程與 Elo/得失球強度處理。
  * `poisson_model.py`：雙卜瓦松訓練與 Dixon-Coles 擬合。
  * `dynamic_simulator.py`：動態 ELO 更新與 2026 賽制路由模擬器（NumPy 加速版）。
* `archive/`：Kaggle 原始數據庫。

---

## 🚀 如何運行

1. **複製本倉庫**：
   ```bash
   git clone https://github.com/wilson94624/FIFA-2026-prediction.git
   cd FIFA-2026-prediction
   ```

2. **安裝環境依賴**：
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **啟動 Notebook**：
   打開 `FIFA_2026_Prediction.ipynb`，點擊 **Restart & Run All** 即可直接復現所有交叉驗證、Dixon-Coles 參數擬合、2022 驗證圖表及 2026 世界盃終極預測奪冠機率分佈圖。
