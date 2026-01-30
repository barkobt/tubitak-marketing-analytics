# 🎯 Marketing Analytics ML Pipeline: From Data Leakage to Prescriptive Optimization

> **End-to-end machine learning system** that transforms 48,000 customer records into actionable marketing strategies, achieving **+26.5% conversion lift** through honest modeling and prescriptive analytics.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)

---

## 🚀 Key Achievements

- ✅ **Detected & eliminated data leakage** that caused false 100% accuracy
- ✅ **Achieved 5.2x lift** over random baseline (F1: 0.15 on 1.3% imbalanced data)
- ✅ **Built prescriptive recommendation engine** with +26.5% expected conversion improvement
- ✅ **Optimized marketing spend** through channel × platform simulation
- ✅ **Production-ready models** with threshold tuning and cross-validation

---

## 📖 Project Overview

### The Challenge
Predicting customer conversion in a **severely imbalanced dataset** (1.3% positive class) while avoiding the trap of data leakage that plagues many ML projects.

### The Solution
A **4-phase analytical pipeline** that:
1. Cleans and validates data integrity
2. Engineers domain-informed features
3. Detects and removes leakage through systematic testing
4. Delivers personalized marketing strategies via simulation

### The Impact
- **218 additional conversions** from optimized channel allocation
- **26.5% improvement** in expected conversion rate
- **Segment-specific strategies** for high-value customer groups

---

## 🏗️ Technical Journey

### **Phase 1: Data Foundation** (`01_eda_and_cleaning.ipynb`)

**Challenge:** 48K records with missing values, outliers, and severe class imbalance

**Actions:**
- Missing data imputation (5% in CTR, PagesPerVisit) via group-based medians
- Outlier detection using IQR and domain constraints (AdSpend < $9K)
- Statistical validation (Chi-square, t-tests) of feature-target relationships

**Key Finding:** 43% conversion rate difference between best (Referral: 1.49%) and worst (SEO: 1.04%) channels

---

### **Phase 2: Feature Engineering** (`02_feature_engineering.ipynb`)

**Challenge:** Raw features had weak predictive power (max correlation: 0.08)

**Engineered 18 features across 5 categories:**

| Category | Examples | Purpose |
|----------|----------|---------|
| **ROI Metrics** | `CPA_Proxy`, `ROI_Proxy`, `Spend_Efficiency` | Marketing efficiency |
| **Engagement** | `Site_Engagement`, `Email_Click_Rate` | User interaction depth |
| **Segmentation** | `Age_Group`, `Income_Tier`, `Loyalty_Tier` | Customer profiling |
| **Interactions** | `AdSpend_x_CTR`, `Income_x_Loyalty` | Non-linear relationships |
| **Channel Intelligence** | `Channel_Performance`, `Is_Best_Channel` | Domain knowledge |

**Output:** 37 total features ready for modeling

---

### **Phase 3: Business Intelligence** (`03_channel_analytics.ipynb`)

**Objective:** Translate data into actionable marketing insights

**Deliverables:**
- **Channel Rankings:** ROI, CPA, and conversion rate comparisons
- **Platform Analysis:** Facebook (1.44%) outperforms YouTube (0.92%)
- **Tool Effectiveness:** Google Ads (1.39%) vs Meta Ads Manager (1.14%)
- **Customer Segmentation:** Age/income distribution by channel

**Business Recommendations:**
1. **HIGH:** Shift 20% budget from SEO → Referral (+56% conversion potential)
2. **MEDIUM:** Prioritize Facebook platform (1.44% conversion)
3. **MEDIUM:** Standardize on Google Ads for consistency

---

### **Phase 4: The Pivot - Data Leakage Discovery** (`04_model_comparison.ipynb`)

#### 🚨 **The Problem**

Initial models achieved **perfect scores** (AUC=1.0, F1=1.0) - impossible for real-world data!

#### 🔍 **Root Cause Analysis**

**Scenario Testing (A-F):**
| Scenario | Features Dropped | Result |
|----------|------------------|--------|
| A | None | AUC=1.0 🚨 LEAK! |
| B | ConversionRate | AUC=1.0 🚨 Still leak! |
| C | + CTR_to_Conversion | AUC=1.0 🚨 Still leak! |
| D | + ROI_Proxy + CPA_Proxy | AUC=0.71 ✅ Clean! |

**Leakage Sources Identified:**
```python
# ❌ LEAKY FEATURES (mathematical dependency on target)
CPA_Proxy = AdSpend / (Conversion + 1)           # Contains target!
ROI_Proxy = (ConversionRate × Income) / AdSpend  # Contains ConversionRate (derived from target)
```

#### ✅ **The Fix: Feature Reconstruction v2**

```python
# ✅ CLEAN FEATURES (no target dependency)
CPA_v2 = AdSpend / (WebsiteVisits + 1)          # Visits, not Conversion
ROI_v2 = (Income × ClickThroughRate) / AdSpend  # CTR, not ConversionRate

# NEW interaction features
Loyalty_per_Purchase = LoyaltyPoints / (PreviousPurchases + 1)
Engagement_Score = (Visits × TimeOnSite × EmailClicks)^(1/3)
```

#### 📊 **Performance Reality Check**

| Metric | Leaky Model | Honest Model | Interpretation |
|--------|-------------|--------------|----------------|
| **F1-Score** | 1.00 | 0.15 | Realistic for 1.3% imbalance |
| **ROC-AUC** | 1.00 | 0.71 | Honest predictive power |
| **Lift vs Random** | N/A | **5.2x** | Meaningful improvement |

**Key Insight:** F1=0.15 is **success**, not failure, for extreme imbalance (industry benchmark: 0.10-0.25)

#### 🎛️ **Threshold Optimization**

Default 0.5 threshold yields F1=0.00 (predicts all negatives). 

**Solution:** Sweep 60 thresholds (0.01-0.60) → Optimal: **0.12-0.15**

---

### **Phase 5: Prescriptive Recommendation System** (`05_recommender_system.ipynb`)

#### 🎯 **Beyond Prediction: What-If Simulation**

**Objective:** Not just "Will they convert?" but **"How can we make them convert?"**

#### ⚙️ **The Engine**

For each customer profile:
1. **Fix** demographic/behavioral features (Age, Income, PreviousPurchases, etc.)
2. **Simulate** all channel × platform combinations (7 channels × 7 platforms = 49 scenarios)
3. **Predict** conversion probability for each combination
4. **Recommend** optimal strategy (highest P(conversion))

```python
# Simulation pseudo-code
for customer in customers:
    current_prob = model.predict(customer.current_strategy)
    best_prob = max([model.predict(customer, channel=c, platform=p) 
                     for c, p in all_combinations])
    lift = best_prob - current_prob
    recommendation = argmax(all_combinations)
```

#### 💰 **Business Impact**

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Expected Conversions** | 821 | 1,039 | **+218 (+26.5%)** |
| **High-Income Segment** | 1.8% | 2.4% | **+33% lift** |
| **Best Strategy** | Display + LinkedIn | - | 20%+ conversion potential |

#### 🎯 **Segment Intelligence**

**High-Value Customers (Income > $80K):**
- **Recommended:** Display + LinkedIn
- **Expected Conversion:** 2.4% (vs 1.8% baseline)

**Young Adults (<30):**
- **Recommended:** Social Media + Instagram
- **Expected Conversion:** 1.9% (vs 1.3% baseline)

---

## 🛠️ Tech Stack

### **Core Libraries**
```
Python 3.10+
├── Data: pandas, numpy
├── ML: scikit-learn, xgboost, lightgbm
├── Imbalance: imbalanced-learn (SMOTE)
├── Validation: StratifiedKFold, threshold_tuning
└── Viz: matplotlib, seaborn
```

### **Key Techniques**
- **SMOTE** for minority class oversampling
- **Threshold Moving** for imbalanced optimization
- **Scenario Testing** for leakage detection
- **Grid Simulation** for prescriptive recommendations

---

## 📂 Project Structure

```
Channel_Analysis/
├── data/
│   ├── marketing_analytics_realistic_48000.csv      # Original
│   ├── marketing_analytics_cleaned.csv              # Phase 1 output
│   └── marketing_analytics_featured.csv             # Phase 2 output
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb                   # Data foundation
│   ├── 02_feature_engineering.ipynb                # Feature creation
│   ├── 03_channel_analytics.ipynb                  # Business intelligence
│   ├── 04_model_comparison.ipynb                   # Leakage detection & fixing
│   └── 05_recommender_system.ipynb                 # Prescriptive engine
├── models/
│   ├── final_model.pkl                              # XGBoost classifier
│   ├── scaler.pkl                                   # StandardScaler
│   └── imputer.pkl                                  # Missing value handler
├── reports/
│   ├── 01_eda/                                      # EDA visualizations
│   ├── 02_features/                                 # Feature distributions
│   ├── 03_channels/                                 # Channel performance
│   ├── 04_models/                                   # Model comparisons
│   └── 05_recommendations/                          # Recommendation outputs
└── README.md
```

---

## 🚀 Quick Start

### **1. Clone & Install**
```bash
git clone https://github.com/your-username/marketing-analytics-ml.git
cd marketing-analytics-ml
pip install -r requirements.txt
```

### **2. Explore Notebooks (Recommended Order)**
```bash
jupyter notebook notebooks/01_eda_and_cleaning.ipynb
# → Continue through 02, 03, 04, 05
```

### **3. Use Trained Model**
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/final_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Predict
customer_data = pd.DataFrame({...})  # Your customer features
scaled_data = scaler.transform(customer_data)
probability = model.predict_proba(scaled_data)[:, 1]
```

### **4. Get Recommendations**
```python
from notebooks.utils import recommend_strategy

# Simulate best channel/platform
recommendation = recommend_strategy(
    customer_profile=customer_data,
    model=model,
    scaler=scaler
)
print(f"Recommended: {recommendation['channel']} + {recommendation['platform']}")
print(f"Expected conversion: {recommendation['probability']:.2%}")
```

---

## 📊 Key Results Summary

### **Model Performance (Honest Baseline)**
- **F1-Score:** 0.15 (5.2x better than random)
- **ROC-AUC:** 0.71 (no leakage)
- **Optimal Threshold:** 0.12-0.15
- **Precision-Recall Balance:** Tuned for business needs

### **Business Impact**
- **+26.5%** expected conversion improvement
- **+218** additional conversions per cohort
- **Segment-specific** strategies (High-income: +33% lift)

### **Technical Innovation**
- **Systematic leakage detection** via A-F scenario testing
- **Feature reconstruction** (v2) without target dependency
- **Prescriptive simulation** engine for actionable insights

---

## 🎓 Lessons Learned

### **1. Data Leakage is Subtle**
Perfect scores (AUC=1.0) are red flags. Always question "too good to be true" results.

### **2. F1=0.15 Can Be Success**
For 1.3% imbalanced data, F1=0.15 represents strong predictive power (5.2x random baseline).

### **3. Threshold Matters**
Default 0.5 threshold is useless for imbalanced data. Always optimize for your metric.

### **4. Prediction → Prescription**
Business value comes from "what to do", not just "what will happen". Simulation-based recommendations bridge this gap.

---

## 👥 Contributors

**Project Lead:** [Your Name]  
**Institution:** TÜBİTAK  
**Timeline:** January 2026  

---

## 📄 License

[Specify license]

---

## 🔗 Links

- **Documentation:** [Link to detailed docs]
- **Blog Post:** [Link to writeup]
- **Presentation:** [Link to slides]

---

**Last Updated:** January 30, 2026  
**Status:** Production-ready model with prescriptive recommendation engine

