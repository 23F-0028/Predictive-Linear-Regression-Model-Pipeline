
# Task 2: Predictive Linear Regression Model Pipeline

## 1. Project Overview
This project implements a supervised machine learning regression pipeline using the California Housing dataset and Scikit-Learn. The objective is to predict median house values using Ridge Regression.

The project includes data loading, preprocessing, model training, evaluation, metric logging, and visualization.

## 2. Objectives
- Load a standard dataset using Scikit-Learn.
- Handle missing values using mean imputation.
- Apply feature scaling to numerical features.
- Perform categorical encoding.
- Train a regularized Ridge Regression model.
- Evaluate the model using R² and RMSE.
- Save evaluation metrics to a CSV file.
- Visualize actual and predicted values.

## 3. Technologies and Libraries
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

## 4. Installation

Install the required libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## 5. Dataset Description

**Dataset:** California Housing

The dataset is loaded using `fetch_california_housing()` from Scikit-Learn.

It contains 20,640 records and eight original numerical input features.

Important features include:
- `MedInc`: Median income
- `HouseAge`: Median house age
- `AveRooms`: Average number of rooms
- `AveBedrms`: Average number of bedrooms
- `Population`: Population
- `AveOccup`: Average household occupancy
- `Latitude`: Geographic latitude
- `Longitude`: Geographic longitude

**Target variable:** `MedHouseVal`

The target represents median house value in units of $100,000.

The original dataset contains numerical features only. Therefore, an `Income_Category` feature is derived from median income to demonstrate categorical encoding.

## 6. Project Workflow

1. Import the required libraries.
2. Load the California Housing dataset.
3. Inspect the dataset and check missing values.
4. Create the categorical `Income_Category` feature.
5. Separate input features and target variable.
6. Split the data into training and testing sets.
7. Apply mean imputation to numerical features.
8. Scale numerical features using StandardScaler.
9. Apply categorical imputation and One-Hot Encoding.
10. Train the Ridge Regression model.
11. Generate predictions on test data.
12. Calculate R² and RMSE.
13. Save the evaluation metrics to a CSV file.
14. Plot actual versus predicted house values.

## 7. Model Description

Ridge Regression is a regularized linear regression algorithm. It uses L2 regularization to reduce the effect of large model coefficients and help control overfitting.

The model is configured with:

- Algorithm: Ridge Regression
- Alpha: 1.0
- Training data: 80%
- Testing data: 20%
- Random state: 42

## 8. Evaluation Metrics

### R² Score
R² measures how well the model explains variation in the target variable. A score closer to 1 generally indicates better performance.

### RMSE
Root Mean Squared Error measures the average magnitude of prediction errors, with larger errors penalized more heavily. A lower RMSE indicates better predictions.

The actual R² and RMSE values are printed after executing the notebook.

## 9. Output Files

The project generates the following output:

- `task2_evaluation_metrics.csv`: Stores the R² and RMSE scores.
- Actual-versus-predicted scatter plot: Visualizes model predictions against actual values.

## 10. How to Run

1. Open the Python notebook in Jupyter Notebook or Google Colab.
2. Install the required libraries.
3. Run all code cells in order.
4. Review the dataset and preprocessing outputs.
5. Check the final R² and RMSE scores.
6. Confirm that the CSV file is saved successfully.
7. Review the actual-versus-predicted graph.

The first dataset download may require an internet connection.

## 11. Conclusion

This project demonstrates a complete supervised regression pipeline using Scikit-Learn. It combines data preprocessing, feature scaling, mean imputation, categorical encoding, Ridge Regression, evaluation metrics, and visualization.

The pipeline helps maintain consistent preprocessing between training and testing data and provides a reusable structure for predictive regression tasks.

## 12. References

- Scikit-Learn California Housing Dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- Scikit-Learn Ridge Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- Scikit-Learn Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- Scikit-Learn ColumnTransformer: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
