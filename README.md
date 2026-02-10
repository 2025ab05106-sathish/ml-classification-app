# ML Assignment 2 — Online Shopper Purchase Prediction

## Problem Statement

Predict whether an online shopping session will result in a **purchase (Revenue)** or not, based on user browsing behavior, session attributes, and temporal features. Compare 6 classification models using 6 evaluation metrics and deploy the solution as a Streamlit web app.

## Dataset Description

- **Name:** Online Shoppers Purchasing Intention Dataset
- **Source:** UCI Machine Learning Repository
- **Link:** https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
- **Samples:** 12,330 sessions
- **Features:** 17 (10 numerical + 7 categorical/integer)
- **Target:** Revenue (Binary — True = Purchase, False = No Purchase)
- **Class Imbalance:** 84.5% No Purchase, 15.5% Purchase

### Feature Details

| Feature | Type | Description |
|---|---|---|
| Administrative | Integer | Number of admin pages visited |
| Administrative_Duration | Float | Time spent on admin pages (seconds) |
| Informational | Integer | Number of info pages visited |
| Informational_Duration | Float | Time spent on info pages (seconds) |
| ProductRelated | Integer | Number of product pages visited |
| ProductRelated_Duration | Float | Time spent on product pages (seconds) |
| BounceRates | Float | Average bounce rate of pages visited |
| ExitRates | Float | Average exit rate of pages visited |
| PageValues | Float | Average page value of pages visited |
| SpecialDay | Float | Closeness to a special day (e.g., Valentine's) |
| Month | Categorical | Month of the session |
| OperatingSystems | Integer | Operating system used |
| Browser | Integer | Browser used |
| Region | Integer | Geographic region |
| TrafficType | Integer | Traffic source type |
| VisitorType | Categorical | Returning, New, or Other |
| Weekend | Boolean | Whether session was on a weekend |

## Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (k=5)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble — Bagging)
6. XGBoost (Ensemble — Boosting)

## Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8779 | 0.8692 | 0.7263 | 0.3403 | 0.4635 | 0.4418 |
| Decision Tree | 0.8467 | 0.7009 | 0.5054 | 0.4895 | 0.4973 | 0.4070 |
| KNN | 0.8674 | 0.7718 | 0.6267 | 0.3560 | 0.4541 | 0.4051 |
| Naive Bayes | 0.7924 | 0.8040 | 0.3981 | 0.6649 | 0.4981 | 0.3971 |
| **Random Forest** | **0.8982** | **0.9172** | **0.7251** | 0.5524 | **0.6270** | **0.5764** |
| XGBoost | 0.8921 | 0.9163 | 0.6824 | **0.5681** | 0.6200 | 0.5609 |

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | High accuracy (0.8779) but low recall (0.3403) — it fails to identify most actual buyers. The linear decision boundary struggles with the non-linear patterns in browsing behavior. High precision (0.7263) means when it predicts a purchase, it's usually right. |
| Decision Tree | Most balanced precision-recall ratio (0.5054/0.4895) among non-ensemble models. However, lowest AUC (0.7009) indicates poor probability calibration. Prone to overfitting on noisy features like browser type and region. |
| KNN | Similar performance to Logistic Regression with slightly lower precision and recall. The distance-based approach is affected by the high dimensionality (17 features) and mixed feature types. AUC of 0.7718 is moderate. |
| Naive Bayes | Lowest accuracy (0.7924) but highest recall among basic models (0.6649). Catches most purchasers but at the cost of many false positives (precision only 0.3981). The feature independence assumption is violated since page duration features are correlated. |
| Random Forest | Best overall model — highest accuracy (0.8982), AUC (0.9172), and MCC (0.5764). The bagging approach with 100 trees effectively handles feature interactions and noisy data. Good balance between precision (0.7251) and recall (0.5524). |
| XGBoost | Very close to Random Forest as the second-best model. Highest recall among high-precision models (0.5681) with AUC of 0.9163. Gradient boosting focuses on hard-to-classify sessions, making it strong on minority class detection. |

**Key Takeaways:**
- Ensemble methods (Random Forest, XGBoost) significantly outperform single models, achieving AUC > 0.91.
- The dataset is imbalanced (84.5% vs 15.5%), making accuracy misleading. MCC provides the most balanced evaluation metric.
- `PageValues` is the most important feature for predicting purchase intent, followed by `ExitRates` and `ProductRelated_Duration`.

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train models (optional — pre-trained models included):
   ```
   cd model
   python train_models.py
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Deployed App

Streamlit Cloud Link: _(add after deployment)_

## Tech Stack

- Python, scikit-learn, XGBoost
- Streamlit for web app
- matplotlib, seaborn for visualizations
