
# Predicting Hospital Readmission for Diabetic Patients

## Project Overview
This project aims to predict hospital readmissions within 30 days for diabetic patients using a medical claims dataset. High readmission rates are a critical issue in healthcare, indicating poor care quality and increasing costs. The Hospital Readmissions Reduction Program (HRRP) penalizes hospitals for excessive readmissions, with diabetic patient readmissions costing $41 billion in 2011. This project addresses two key objectives:

1. **Identify the strongest predictors** of hospital readmission for diabetic patients.
2. **Develop an accurate predictive model** using a limited set of features.

The project employs machine learning techniques, including Decision Trees, XGBoost, and LightGBM, to model readmission risk and evaluate predictor importance. The final model achieves high recall and reasonable precision, making it valuable for healthcare applications.

## Dataset Description
The dataset, sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008), contains medical claims data for diabetic patients across 130 US hospitals from 1999–2008. It includes 101,766 encounters with 50 features.

### Key Variables
| **Variable**                | **Description**                                      | **Values/Format**                          |
|-----------------------------|-----------------------------------------------------|--------------------------------------------|
| Encounter ID                | Unique encounter identifier                         | Numeric                                    |
| Patient Number              | Unique patient identifier                           | Numeric                                    |
| Race                        | Patient’s race                                      | Caucasian, Asian, African American, etc.   |
| Gender                      | Patient’s gender                                    | Male, Female, Unknown/Invalid             |
| Age                         | Patient’s age group                                 | 10-year intervals (e.g., [0-10), [90-100))|
| Admission Type              | Type of admission                                   | Emergency, Urgent, Elective, etc.          |
| Time in Hospital            | Days from admission to discharge                    | Integer                                    |
| Number of Lab Procedures    | Lab tests performed                                 | Numeric                                    |
| Number of Medications       | Distinct medications given                          | Numeric                                    |
| Number of Diagnoses         | Total diagnoses recorded                            | Numeric                                    |
| Glucose Serum Test Result   | Blood glucose level                                 | >200, >300, Normal, None                  |
| A1c Test Result             | A1c test outcome                                    | >8%, >7%, Normal, None                    |
| Diabetes Medications        | Diabetic medication prescribed                      | Yes, No                                    |
| Readmitted                  | Time to readmission                                 | <30 days, >30 days, No                    |

### Data Preprocessing
- **Missing Values**: Replaced '?' with 'Unknown' or 'Missing' for categorical variables (e.g., race, diagnoses). Dropped columns with excessive missing data (e.g., weight, payer_code).
- **Encoding**: Converted categorical variables (e.g., gender, medications) to numeric. One-hot encoded key categorical features (e.g., medical specialty, diagnoses).
- **Feature Engineering**: Created log-transformed features, ratio features (e.g., meds_per_diag), and interaction terms to capture complex relationships.
- **Aggregation**: Grouped data by patient to compute mean/sum/max of features, reducing encounter-level noise.

## Methodology
The project follows a structured approach to analyze the dataset, identify predictors, and build predictive models.

### Steps
1. **Data Exploration**:
   - Analyzed dataset for missing values, distributions, and imbalances.
   - Visualized relationships between features (e.g., time in hospital, number of medications) and readmission.

2. **Feature Engineering**:
   - Created new features like service_utilization (sum of outpatient, emergency, and inpatient visits).
   - Grouped ICD-9 codes into categories (e.g., Diabetes, Circulatory).
   - Applied log transformations to skewed features.

3. **Modeling**:
   - **Feature Selection**: Used Recursive Feature Elimination (RFE) with RandomForest to select the top 30 features.
   - **Data Balancing**: Applied ADASYN to address class imbalance (88% non-readmitted vs. 12% readmitted).
   - **Models Tested**:
     - Decision Tree
     - XGBoost
     - LightGBM
   - **Hyperparameter Tuning**: Used GridSearchCV with cross-validation to optimize model performance.
   - **Evaluation Metrics**: Focused on recall (to capture readmissions), precision, F1-score, ROC-AUC, and PR-AUC.

4. **Final Model**:
   - Selected XGBoost with tuned hyperparameters for its balance of performance and interpretability.
   - Adjusted prediction threshold to prioritize recall ≥ 0.70 while maximizing precision.

### Models and Hyperparameters
| **Model**       | **Best Hyperparameters**                                                                 | **Scoring Metric** |
|-----------------|-----------------------------------------------------------------------------------------|---------------------|
| Decision Tree   | `criterion='gini', max_depth=7, min_samples_leaf=2, min_samples_split=5`                | F1-score            |
| XGBoost         | `learning_rate=0.05, max_depth=5, n_estimators=100, scale_pos_weight=3, subsample=0.8`  | F1-score            |
| LightGBM        | `learning_rate=0.05, max_depth=5, n_estimators=100, num_leaves=20`                     | Recall              |

## Results
The final XGBoost model was evaluated on a test set (20% of data) with a focus on identifying patients at risk of readmission within 30 days.

### Performance Metrics
| **Metric**          | **Value** | **Description**                                                                 |
|---------------------|-----------|--------------------------------------------------------------------------------|
| **Recall (Class 1)**| 0.70      | Captures 70% of patients readmitted within 30 days, critical for intervention.  |
| **Precision (Class 1)** | 0.45  | 45% of predicted readmissions are correct, balancing false positives.          |
| **F1-Score (Class 1)** | 0.55 | Harmonic mean of precision and recall, indicating balanced performance.         |
| **ROC-AUC**         | 0.859     | Strong discrimination between readmitted and non-readmitted patients.           |
| **PR-AUC**          | 0.533     | Good performance in the precision-recall space, given class imbalance.         |
| **Accuracy**        | 0.85      | Overall correct predictions, though less critical due to imbalance.             |

### Confusion Matrix (Threshold = 0.678)
| **Actual\Predicted** | **Non-Readmitted (0)** | **Readmitted (1)** |
|-----------------------|------------------------|--------------------|
| **Non-Readmitted (0)** | 10,808                | 1,522             |
| **Readmitted (1)**     | 527                   | 1,232             |

### Key Predictors
The strongest predictors of readmission, based on feature importance from the XGBoost model, include:
1. **Number of Inpatient Visits**: Higher prior inpatient visits strongly correlate with readmission risk.
2. **Service Utilization**: Total healthcare interactions (outpatient, emergency, inpatient) indicate patient complexity.
3. **Time in Hospital**: Longer hospital stays suggest severe conditions, increasing readmission likelihood.
4. **Number of Medications**: More medications reflect complex treatment regimens, linked to higher risk.
5. **Change in Medications**: Adjustments in diabetic medications signal unstable disease management.

### Visualizations
Visualizations confirmed key relationships:
- **Time in Hospital vs. Readmission**: Readmitted patients have slightly longer hospital stays (KDE plot).
- **Age vs. Readmission**: Older patients (70–90 years) have higher readmission rates.
- **Service Utilization**: Patients with higher utilization are more likely to be readmitted.

## Industry Standards and Achievements
### Industry Context
In healthcare, predicting hospital readmissions is critical for:
- **Cost Reduction**: The HRRP penalizes hospitals for excessive readmissions, with diabetic readmissions costing billions annually.
- **Patient Outcomes**: Early identification of high-risk patients enables targeted interventions (e.g., follow-up care, medication adjustments).
- **Regulatory Compliance**: Hospitals must meet CMS (Centers for Medicare & Medicaid Services) standards for readmission rates.

**Industry Benchmarks**:
- **Recall**: High recall (≥0.70) is prioritized to minimize missed readmissions, as false negatives can lead to patient harm.
- **Precision**: Precision ≥0.50 is desirable to reduce unnecessary interventions, but 0.40–0.50 is acceptable in imbalanced datasets.
- **ROC-AUC**: Values ≥0.80 indicate strong model discrimination.
- **PR-AUC**: Values ≥0.50 are good for imbalanced datasets like this one.

### Achievements
| **Aspect**                  | **Achievement**                                                                 | **Industry Relevance**                                                                 |
|-----------------------------|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| **Recall**                  | Achieved 0.70, capturing 70% of readmissions.                                  | Meets industry need to identify high-risk patients, reducing missed interventions.     |
| **Precision**               | Achieved 0.45, close to the 0.50 benchmark.                                    | Balances resource allocation, though slightly below ideal due to class imbalance.      |
| **ROC-AUC**                 | Achieved 0.859, exceeding the 0.80 benchmark.                                  | Indicates robust model performance, suitable for clinical decision support.            |
| **PR-AUC**                  | Achieved 0.533, meeting the ≥0.50 benchmark.                                   | Strong performance in imbalanced settings, relevant for rare events like readmissions. |
| **Key Predictors**          | Identified actionable predictors (e.g., inpatient visits, medication changes). | Enables hospitals to focus interventions on high-risk factors, improving outcomes.     |
| **Model Efficiency**        | Used limited features (30) with high performance.                              | Aligns with industry need for interpretable, scalable models in resource-constrained settings. |

### Domain Impact
- **Clinical Utility**: The model identifies 70% of patients at risk of readmission, enabling hospitals to implement targeted follow-up care, such as post-discharge monitoring or diabetes management programs.
- **Cost Savings**: By reducing readmissions, hospitals can avoid HRRP penalties and lower the $41 billion cost burden associated with diabetic readmissions.
- **Scalability**: The use of a limited feature set and interpretable predictors (e.g., service utilization) makes the model feasible for integration into electronic health record (EHR) systems.
- **Patient-Centric Care**: Focusing on predictors like medication changes highlights the need for better disease management, improving patient outcomes.

## Installation and Usage
### Prerequisites
- Python 3.8+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `imblearn`, `seaborn`, `matplotlib`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Vraj-Data-Scientist/prediction-on-hospital-readmission.git
   cd prediction-on-hospital-readmission
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) and place it in the project directory as `diabetic_data.csv`.

### Running the Notebook
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `prediction-on-hospital-readmission.ipynb` and run all cells to reproduce the analysis.

## Future Work
- **Incorporate Additional Features**: Include social determinants of health (e.g., income, access to care) to improve model performance.
- **Real-Time Prediction**: Integrate the model into EHR systems for real-time risk scoring.
- **Threshold Optimization**: Explore ensemble methods to achieve precision ≥0.50 while maintaining recall ≥0.70.
- **External Validation**: Test the model on newer datasets to ensure generalizability.

## Acknowledgments
- Dataset provided by the UCI Machine Learning Repository.
- Inspired by the Hospital Readmissions Reduction Program and the need to improve diabetic patient outcomes.
