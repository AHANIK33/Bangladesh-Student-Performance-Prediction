# Bangladesh Student Performance Prediction

Predicts a student's HSC (Higher Secondary Certificate) GPA from demographic, family, and academic background features using regression models.

## Overview

Using the [Bangladesh Student Performance dataset](#dataset), this project builds a preprocessing + regression pipeline to predict `hsc_result` (HSC GPA) from features like gender, address, family background, parental education/job, tuition fee, and prior SSC result.

## Dataset

- **File:** `bangladesh_student_performance_updated.csv`
- **Rows:** 2,018 students, 16 columns
- **Target:** `hsc_result` (HSC GPA, scale 2.0–5.0)
- **Dropped column:** `date` (constant, not predictive)
- **Features used:** `gender`, `age`, `address`, `famsize`, `Pstatus`, `M_Edu`, `F_Edu`, `M_Job`, `F_Job`, `relationship`, `smoker`, `tuition_fee`, `time_friends`, `ssc_result`

See the dataset's own README for full column descriptions.

## Approach

1. **Preprocessing** (via `ColumnTransformer`):
   - **Nominal categorical columns** (`gender`, `address`, `famsize`, `Pstatus`, `M_Job`, `F_Job`, `relationship`, `smoker`) → most-frequent imputation + one-hot encoding
   - **Numerical columns** (`age`, `tuition_fee`, `ssc_result`) → mean imputation + standard scaling
   - Remaining columns (`M_Edu`, `F_Edu`, `time_friends`) passed through unchanged
2. **Split:** 80% train / 20% test (`random_state=42`)
3. **Models trained:**
   - Linear Regression
   - SGD Regressor
4. **Evaluation metrics:** MSE, RMSE, MAE, R²

## Results (Test Set)

| Model | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| **Linear Regression** | 0.0203 | 0.1425 | 0.1114 | **0.9459** |
| SGD Regressor | 0.0204 | 0.1430 | 0.1126 | 0.9455 |

Both models perform very similarly, explaining ~95% of the variance in HSC results. Linear Regression is marginally better and simpler to interpret, making it the preferred model here.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. Place `bangladesh_student_performance_updated.csv` in your working directory (update the file path in the notebook if not using Colab).
2. Run `Bangladesh_Student_Performance_update.ipynb` top to bottom.
3. To predict a new student's HSC result:

```python
new_student = pd.DataFrame({
    'gender': ['Male'], 'address': ['Urban'], 'famsize': ['GT3'],
    'Pstatus': ['Together'], 'M_Job': ['Teacher'], 'F_Job': ['Business'],
    'relationship': ['Good'], 'smoker': ['No'], 'age': [18],
    'tuition_fee': [5000], 'ssc_result': [4.80],
    'F_Edu': [12], 'M_Edu': [14], 'time_friends': [3]
})

prediction = lr_pipe.predict(new_student)
print(prediction)
```

## Project Structure

```
.
├── Bangladesh_Student_Performance_update.ipynb   # Preprocessing, training, evaluation
├── bangladesh_student_performance_updated.csv    # Dataset (add your own)
└── README.md
```

## Future Improvements

- Try regularized models (Ridge, Lasso) or tree-based regressors (Random Forest, Gradient Boosting)
- Hyperparameter tuning for `SGDRegressor` (learning rate, penalty)
- Feature importance / coefficient analysis to identify the strongest predictors of HSC performance
- Save the trained pipeline with `joblib` for reuse without retraining
