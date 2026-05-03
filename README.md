# AI Developer Productivity Analysis

**Statistical Learning & Regularization Techniques**

Analysis of AI developer cognitive load and productivity factors using OLS regression, regularization methods (LASSO, Ridge, Elastic Net), and multicollinearity diagnosis.

---

## 📋 Project Overview

This project investigates the relationship between developer work habits and cognitive load through rigorous statistical modeling. The study applies regression analysis with emphasis on handling multicollinearity and feature selection through regularization techniques.

**Course:** Statistical Models & Statistical Learning  
**Program:** M.Sc. Computer Engineering (AI & Machine Learning track)  
**University:** Università della Calabria

---

## 📊 Dataset

**Source:** [AI Developer Productivity Dataset](https://www.kaggle.com/datasets/atharvasoundankar/ai-developer-productivity-dataset)

**Size:** 500 observations (daily work sessions)

### Features (8 predictors)
- `hours_coding` — Total hours spent writing code
- `coffee_intake_mg` — Caffeine consumption in milligrams
- `distractions` — Number of interruptions during work
- `sleep_hours` — Hours of sleep
- `commits` — Code commits performed
- `bugs_reported` — Bugs identified
- `ai_usage_hours` — Time spent using AI tools
- `task_success` — Binary indicator of task completion (0/1)

### Target Variable
- `cognitive_load` — Perceived/measured cognitive load (scale 1–10)

---

## 🔍 Methodology

### 1. **Exploratory Data Analysis (EDA)**
- Descriptive statistics and null value assessment
- Distribution analysis of target variable (cognitive_load)
- Correlation matrix and pairwise relationships
- Identification of potential multicollinearity issues

### 2. **OLS Regression Models**
Multiple model specifications were tested:

| Model | Formula | R² | Condition Number |
|-------|---------|-----|------------------|
| **Lin-Lin (full)** | $CL = \beta_0 + \sum \beta_i X_i + \epsilon$ | 0.731 | 3,570 |
| **Lin-Lin (no coffee)** | Excludes `coffee_intake_mg` | 0.731 | **70.0** |
| **Lin-Lin (interactions)** | Includes $HC \times SH$ interaction | 0.729 | 203 |
| **Log-Lin** | $\ln(CL) = \beta_0 + \sum \beta_i X_i + \epsilon$ | 0.689 | 3,570 |
| **Log-Log** | $\ln(CL) = \beta_0 + \sum \beta_i \ln(X_i) + \epsilon$ | 0.662 | 177 |

### 3. **Multicollinearity Diagnosis**
- **Correlation analysis:** Identified near-perfect correlation (ρ = 0.89) between `hours_coding` and `coffee_intake_mg`
- **VIF metrics:** 
  - `coffee_intake_mg` (VIF = 55.74)
  - `hours_coding` (VIF = 49.62)
- **Condition Number:** Initial value 3,570 → 70 after removing `coffee_intake_mg`

### 4. **Model Diagnostics**
- ✅ **Normality (Q-Q plot):** Residuals approximately normal
- ✅ **Homoscedasticity (Breusch-Pagan test):** p-value = 0.25 (no heteroscedasticity)
- ✅ **Autocorrelation (Durbin-Watson):** DW ≈ 1.84 (no first-order correlation)

### 5. **Regularization Techniques**
Applied three regularization approaches with 5-fold cross-validation:

$L_{\text{LASSO}}(\beta, \lambda) = (\mathbf{y} - \mathbf{X}\beta)^T(\mathbf{y} - \mathbf{X}\beta) + \lambda \sum_{j=1}^{p} |\beta_j|$

$L_{\text{Ridge}}(\beta, \lambda) = (\mathbf{y} - \mathbf{X}\beta)^T(\mathbf{y} - \mathbf{X}\beta) + \lambda \sum_{j=1}^{p} \beta_j^2$

$L_{\text{ENet}}(\beta, \lambda_1, \lambda_2) = (\mathbf{y} - \mathbf{X}\beta)^T(\mathbf{y} - \mathbf{X}\beta) + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2$

---

## 📈 Key Results

### **Selected Model: LASSO**

| Metric | Value |
|--------|-------|
| **R² (Test Set)** | 0.6918 |
| **AIC** | -5.07 |
| **BIC** | 10.56 |
| **Condition Number** | **1.71** |
| **Variables Retained** | 5 (out of 8) |

### Variables Eliminated by LASSO
- `hours_coding` (redundant with other predictors)
- `coffee_intake_mg` (multicollinear with hours_coding)
- `bugs_reported` (low predictive power)

### Final Model (5 predictors)
1. **distractions** (t = 17.97, p < 0.001)
2. **sleep_hours** (t = -28.23, p < 0.001)
3. **commits** (t = 1.99, p = 0.047)
4. **ai_usage_hours** (t = 0.97, p = 0.332)
5. **task_success** (t = -1.61, p = 0.108)

### Key Insights
- **Sleep is the strongest predictor** of reduced cognitive load
- **Distractions significantly increase** perceived cognitive burden
- **Commits** show marginal but significant effect
- LASSO reduced condition number from **3,570 → 1.71**, enabling stable coefficient interpretation

---

---

## 📊 Key Visualizations

The analysis includes:

1. **Distribution of cognitive load** (histogram + density)
2. **Correlation matrix** (heatmap with VIF analysis)
3. **Scatter plots** (pairwise relationships)
4. **Q-Q plot** (residual normality)
5. **Residuals vs. Fitted values** (homoscedasticity check)
6. **Regularization paths** (LASSO, Ridge, Elastic Net coefficient shrinkage)
7. **Observed vs. Predicted** (model performance on test set)

---

## 📌 Statistical Findings

### Multicollinearity Resolution
- **Before regularization:** Condition number = 3,570 (severe multicollinearity)
- **After LASSO:** Condition number = 1.71 (near-orthogonal predictors)
- **Impact:** Coefficient estimates become stable and interpretable

### Model Performance
- **OLS vs. LASSO:** Similar R² (0.731 → 0.692), but LASSO is more parsimonious and generalizable
- **Test set R²:** 0.692 indicates good out-of-sample performance
- **Variable selection:** LASSO automatically identified 3 redundant features

### Diagnostic Compliance
| Test | Result | Interpretation |
|------|--------|-----------------|
| Normality (Shapiro-Wilk) | ✅ Passed | Residuals ~ N(0, σ²) |
| Homoscedasticity (BP test) | ✅ Passed (p=0.25) | Constant variance |
| Autocorrelation (DW) | ✅ Passed (1.84) | Independent errors |
| Multicollinearity (VIF) | ✅ Resolved | All VIF < 5 post-LASSO |

---

## 📝 Full Report

Complete analysis available in: **`relazione_statistica.pdf`**
