# credit-risk-classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and data for a credit risk classification project, focusing on predicting loan health using machine learning techniques. The primary goal is to develop a model that can accurately identify high-risk loans, enabling proactive risk management for lending institutions.

## Repository Structure

```
credit-risk-classification/  (Root Directory)
├── README.md                 (This file)
├── credit_risk_classification_ipynb_Explanation     (Explanation of the actual code used in the analysis)
├── report-credit_risk_classification.ipynb     (Report Jupyter Notebook)
└── Credit_Risk/             (Subdirectory)
    ├── credit_risk_classification.ipynb (Main Code Jupyter Notebook)
    └── lending_data.csv      (Dataset used for analysis)
```

## Project Overview

This project tackles a crucial business problem: predicting the risk associated with lending money. By accurately classifying loans as either "healthy" or "high-risk," lending institutions can make better-informed decisions, minimize potential financial losses, and optimize their lending strategies. 

The primary tool used in this project is the Logistic Regression model, a linear model that is readily applicable to binary classification tasks. Before utilizing the model, a data scaling method is implemented using StandardScaler and the models are run both with and without the data scaling to assess what effects the scaling has on the reliability, validity, and efficacy of the model to provide accurate feedback to the user.

## Files

*   **`README.md`:** This file, providing an overview of the project, repository structure, and instructions for use.
*   **`credit_risk_classification_ipynb_Explanation.pdf`:** A PDF file that details the approach embodied in the code that is used for the analysis.
*   **`report-credit_risk_classification.ipynb`:** A Jupyter Notebook for the project's report.  It contains the key sections for a complete analysis of the code and the results.  Please view for detailed analysis.
*   **`Credit_Risk/credit_risk_classification.ipynb`:** A Jupyter Notebook containing the main code for the project, including data loading, preprocessing, model training, and evaluation. Based on the *Scikit-learn* and *Pandas* libraries.
*   **`Credit_Risk/lending_data.csv`:** The dataset used for training and evaluating the credit risk classification model.

## Code Implementation

**Notebook:`report-credit_risk_classification.ipynb`**

**Overview of the Analysis**

The purpose of this analysis is to build and evaluate a machine learning model that can predict
loan risk. Specifically, the model is trained to classify loans as either healthy (low risk of default)
or high-risk. The financial information used includes loan size, interest rate, borrower income,
debt-to-income ratio, number of accounts, derogatory marks, and total debt. Accurately
identifying high-risk loans is crucial for mitigating potential financial losses and informing
lending decisions. We used Logistic Regression, initially without scaling, and then with data
scaling using StandardScaler, as part of our modeling approach.  Please view this file for more details.

**Notebook:`credit_risk_classification.ipynb`**

**Steps**
* Import the *Numpy*, *Pandas*, and *Scikit-learn* methods
* Split the dataset into testing and training data.

**Creating Machine Learning Models**
* Load the *lending_data.csv* dataset.
* Divide into features and target variables.
* Divide into test and training data.

**Data Scaling and Preprocessing**
* Standardize the data to run using the *StandardScaler* method.

**Analysis, Running Models and Evaluation**
* *Run* the Logistic Regression model on the dataset.
* *Evaluate* the models *both* with and without scaling and preprocessing by displaying a matrix and other metrics.

**Results**

*   **Machine Learning Model 1: Logistic Regression (Without Data Scaling)**

    *   **Accuracy:** 0.99
    *   **Precision (Healthy Loans):** 1.00
    *   **Recall (Healthy Loans):** 0.99
    *   **Precision (High-Risk Loans):** 0.84
    *   **Recall (High-Risk Loans):** 0.94

*   **Machine Learning Model 2: Logistic Regression (With Data Scaling)**

    *   **Accuracy:** 0.99
    *   **Precision (Healthy Loans):** 1.00
    *   **Recall (Healthy Loans):** 0.99
    *   **Precision (High-Risk Loans):** 0.84
    *   **Recall (High-Risk Loans):** 0.99

**Summary**

The Logistic Regression model, both before and after data scaling, was used to classify loans. Before scaling, the model exhibited high accuracy and perfect precision and recall for healthy loans. However, the precision and recall for high-risk loans were lower with 0.84 and 0.94 respectively, suggesting some difficulty in accurately identifying all high-risk loans and some misclassification of healthy loans as high-risk.

After scaling the data using StandardScaler, there was a small improvement to the model's ability to classify the loan. The Logistic Regression model demonstrates strong performance in predicting loan risk. The high accuracy (0.99) indicates that the model is generally correct in its classifications. The model excels at identifying healthy loans, achieving nearly perfect precision and recall. While the precision for high-risk loans is lower (0.84), the recall is very high (0.99), meaning the model effectively identifies nearly all high-risk loans.

Comparison of Results:

Scaling the data had a notable impact on the model's performance, specifically for the Recall of High-Risk loans, increasing it from 0.94 to 0.99. This improvement is likely because the scaling process addresses issues with features being on different scales, allowing the optimization algorithm to converge more effectively. There was no significant change to the overall accuracy of the model or the Recall and Percision of Healthy Loans, however this is not a critical measure. The improved High-Risk Recall indicates a stronger ability to identify high-risk loans.

**Recommendation:**

I recommend the scaled model for use by the company with caution. Its extremely high recall for high-risk loans (0.99) is critical from a business perspective, as it minimizes the risk of missing potentially defaulting loans, and maintains a high degree of Precision (0.84) compared to the unscaled model. The trade-off is that there are some healthy loans misclassified as high-risk, the cost of which will depend on the details of the company's loan decisions. The cost of misclassifying a high-risk loan greatly outweighs the cost of missing the classification. I recommend continuous monitoring and refinement of the model to further improve its precision for high-risk loans and minimize the misclassification of healthy loans. Due to the scaling resulting in an improvement in the High-Risk Recall, I recommend only the scaled model for use by the company.

## Dependencies

This project requires the following Python libraries:

*   pandas
*   numpy
*   scikit-learn
*   pathlib

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn pathlib
```

## Usage

1.  Clone the repository:
    ```bash
    git clone https://github.com/AlexGerwer/credit-risk-classification.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd credit-risk-classification
    ```
3.  (Optional) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate.bat # On Windows
    ```
4.  Install the dependencies:
    install directly with *pip install pandas numpy scikit-learn pathlib*
    
5.  Open and run the `Credit_Risk/credit_risk_classification.ipynb` Jupyter Notebook to perform the analysis.

## Contributing

Contributions to this project are welcome! Please feel free to submit pull requests with bug fixes, improvements, or new features.

## License

This project is licensed under the MIT License.


