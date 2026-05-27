# 🏠 California Housing Price Prediction

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-red.svg)](https://xgboost.ai/)

## 📊 Project Overview
This project predicts California housing prices using advanced machine learning techniques. The final stacking ensemble achieves **R² = 0.851**, explaining 85% of price variation.

### Key Results
| Model | R² Score | RMSE |
|-------|----------|------|
| Linear Regression | 0.576 | $74,558 |
| Random Forest | 0.774 | $54,472 |
| XGBoost (Tuned) | 0.841 | $54,222 |
| **Stacking Ensemble** | **0.851** | **$53,000** |

## 🛠️ Technical Approach
1. **Exploratory Data Analysis** - Correlation analysis, distribution checks
2. **Feature Engineering** - Scaling, encoding categorical variables
3. **Model Tuning** - Optuna Bayesian optimization (50 trials)
4. **Ensemble Methods** - Voting + Stacking with Ridge meta-learner

## 🚀 Quick Start
```bash
git clone https://github.com/yourusername/house-price-prediction
cd house-price-prediction
pip install -r requirements.txt
python train_model.py