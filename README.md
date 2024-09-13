# Profit Prediction for Fortune 500 Companies

## Folder Info
- Final Data folder name -> Raw_Data
- Final ML Models folder name -> ML_Model
- Final progress folder name -> Report
- Final visualizations folder name -> Visualizations

## Overview
This project aims to predict the profit of Fortune 500 companies using various machine learning models. By leveraging a dataset from Kaggle, we employed techniques such as data preprocessing, feature engineering, and model evaluation to build and test multiple predictive models.

## Dataset
The dataset includes financial metrics such as revenue, expenses, assets, and liabilities for Fortune 500 companies. These variables were used to train machine learning models to predict profitability.

## Project Structure
- `Data/`: Contains the raw and processed datasets used for the project.
- `Notebooks/`: Jupyter notebooks for data analysis, preprocessing, and modeling.
- `Models/`: Includes different machine learning models trained and their performance metrics.
- `Reports/`: Final and progress reports documenting the methodology, challenges, and results.
  - [Final Report](https://github.com/yshah33/Profit_Prediction_Fortune500_Companies/blob/main/Reports/CS_418-%20Group_11-%20Final_Report.docx)
  - [Progress Report](https://github.com/yshah33/Profit_Prediction_Fortune500_Companies/blob/main/Reports/CS%20418%20-%20Group%2011%20-%20Progress%20Report_.docx)

## Technologies Used
- **Python** for scripting and data analysis
- **Pandas** and **NumPy** for data manipulation
- **scikit-learn** for machine learning algorithms
- **Matplotlib** and **Seaborn** for data visualization
- **Jupyter Notebook** for interactive coding and documentation

## Machine Learning Models
1. **Linear Regression**: A baseline model to understand relationships between financial variables and profitability.
2. **Decision Trees**: A more complex model to capture non-linear relationships.
3. **Random Forest**: An ensemble model that improves prediction by combining multiple decision trees.

## Key Features
- **Data Preprocessing**: Handled missing data, normalized financial metrics, and performed feature selection to enhance model performance.
- **Model Evaluation**: Models were evaluated using R², Mean Absolute Error (MAE), and Mean Squared Error (MSE).
- **Predictive Insights**: The models provide insights into which financial metrics most influence company profitability.

## Results
The Random Forest model provided the best performance with an R² score of 0.85, indicating a strong correlation between the predicted and actual profits.

## Future Work
- Incorporating more financial data from other sources.
- Exploring advanced models like Gradient Boosting and Neural Networks.
- Automating hyperparameter tuning to further optimize model performance.

## Usage
Clone the repository and navigate to the `Notebooks/` folder to explore the model development process.

```bash
git clone https://github.com/yshah33/Profit_Prediction_Fortune500_Companies.git
cd Profit_Prediction_Fortune500_Companies/Notebooks
