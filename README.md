## INFO6105 Data Science Engineering Methods and Tools

## Overview
This repository contains assignments for the course INFO6105 - Data Science Engineering Methods and Tools. Each Jupyter Notebook is a stand-alone 
assignment that covers various topics and techniques in data science, including data analysis, machine learning models, data visualization, and more.

## Context
The data is the price history and trading volumes of the fifty stocks in the index NIFTY 50 from NSE (National Stock Exchange) India. All datasets are at a 
day-level with pricing and trading values split across .cvs files for each stock along with a metadata file with some macro-information about the stocks 
itself. The data in the dataset spans from 1st January, 2000 to 30th April, 2021.

### Metadata
Date - Trade Data: Represents the date of the trading data, indicating when the stock market activity occurred.

Symbol - Name of Stock: Refers to the unique code or symbol assigned to a particular stock. It is essentially the shorthand representation of the company's 
name on the stock exchange.

Series - Type of Security: Specifies the type of security, which could include different classes of financial instruments such as equity shares, preference 
shares, or other financial products.

Prev Close - Previous Day's Closing Price: Indicates the closing price of the stock on the previous trading day.

Open - Opening Price for the Day: Represents the initial price at which a stock is traded on a given day.

High - Highest Price for the Day: Denotes the highest trading price reached by the stock during the trading day.

Low - Lowest Price for the Day: Represents the lowest trading price reached by the stock during the trading day.

Last - Last Trade Price: Refers to the price at which the last trade was executed.

Close - Closing Price: Indicates the final trading price of the stock at the end of the trading day.

VWAP - Volume-Weighted Average Price: VWAP is a ratio of the cumulative share price to the cumulative volume traded over a given time period. It provides 
insight into the average price at which a stock is traded, weighted by the volume of trades.

Volume - Volume Traded for the Day: Represents the total number of shares or contracts traded during a specific time period, typically a trading day.

Turnover - Turnover Ratio: The turnover ratio is the ratio of sellers to buyers of a stock. It helps in understanding the market activity and liquidity.

Trades - Number of Trades: Indicates the total number of trades (buy and sell transactions) that occurred during the trading day.

Deliverable Volume - Amount of Deliverable Volume: Represents the volume of shares that were actually delivered (transferred) as opposed to being traded 
intraday.

% Deliverable - Percentage of Deliverable Shares: Indicates the percentage of shares that were delivered out of the total traded volume.

Note: All the prices are denoted in Indian Rupees (INR), as mentioned at the end of the description. This dataset provides comprehensive information about the 
trading activity of various stocks, allowing users to analyze and understand the market trends and stock behavior.

## Assignments

### Assignment 1: ML Data Cleaning and Feature Selection
Description: The dataset features 1 datetime, 2 categorical, and 12 numerical columns, with missing values in 'Trades', 'Deliverable Volume', and 
'%Deliverable' filled with zeros for justified reasons. It exhibits right skewness and outliers in numerical columns. Key predictors identified for 'Close' 
include 'Prev Close', 'Open', 'High', 'Low', 'Last', and 'VWAP'. Decision Tree Regressor shows high performance, but risks overfitting. Analysis shows 
significant multicollinearity among predictors and extreme right-skewness in distributions. Model performance improves upon outlier removal, showing lower 
error metrics. Imputation tests post-random data removal indicate stable bias, variance, and residual error. Overall, the dataset analysis focused on managing 
missing data, feature importance, multicollinearity, outlier impacts, and model evaluation in this assignment.

Notebook: Assignment1_AnushaGali_INFO6105_latest.ipynb

### Assignment 2: Auto ML
Description: In this assignment, we perform AutoML using H2O library on the Stock dataset. We used various algorithms to predict the close price of stock on 
its features. After evaluating the models using various metrics, we found that the H2O AutoML model with regularization and without regularization provided a 
similar performance. The most significant predictor variables in the final model were 'High', 'Last', 'Low', 'VWAP'. We also found that there was no violation 
of model assumptions, and multicollinearity was not a significant issue. Overall, in all the models returned by AutoML we observed that there was similar
performance and the MAE was set as the deciding factor in the best model given that all the models make sense.

Notebook: Assignment2_Anusha_Gali_INFO6105_Latest.ipynb

### Assignment 3: Model Interpretability
Description: In the assignment we performed SHAP (SHapley Additive exPlanations) analysis on three models, revealing how individual features influence the 
predicted outcomes. This analysis improved interpretability by detailing feature importance and impact, providing actionable insights for model optimization.
In the linear model, an inverse relationship was observed between SHAP values and feature values, challenging traditional assumptions and highlighting unique 
data dynamics. Both the AutoML's best model and the Random Forest model, being tree-based, showed similar SHAP plots, indicating consistent feature usage and
decision-making processes. This consistency across tree-based models boosts confidence in their reliability and interpretability, aiding in informed 
decision-making and model assessment. Overall, SHAP analysis offered crucial insights into feature relevance and model behaviors, enhancing our ability to 
make more precise modeling decisions.

Notebook: Assignment3_AnushaGali_INFO6105_Latest.ipynb

### Assignment 4: Combine Data Cleaning, Feature Selection, Modeling, and Interpretability into one report
Description: This assignment is a combination of all the above assignments

Notebook: Assignment4_AnushaGali_INFO6105.ipynb

### Assignment 5: Neural Network Type Classification | TMNIST
Description: The Convolutional Neural Network (CNN) model architecture consists of multiple convolutional layers followed by max-pooling layers for feature 
extraction and spatial reduction. It also includes dropout layers for regularization and fully connected dense layers for classification. The model is trained
using the Adam optimizer with categorical crossentropy loss and accuracy as the evaluation metric. Early stopping is employed to prevent overfitting by
monitoring the validation loss. During training, the model achieves significant improvements in both training and validation accuracy, indicating effective 
learning. The validation loss decreases consistently, suggesting good generalization to unseen data. The model achieves a high test accuracy of 94.81%, 
demonstrating its ability to accurately classify unseen samples. The plots of training and validation accuracy/loss show typical behavior for a well-trained 
model. Both training and validation accuracy increase over epochs, while the loss decreases. The validation loss flattens out, indicating good generalization.
Overall, the CNN model demonstrates strong performance in both training and evaluation phases, achieving high accuracy and effectively generalizing to unseen 
data. The training history plots indicate successful learning without overfitting. This suggests that the model has effectively learned meaningful patterns 
from the data and can reliably classify images into their respective classes.

Notebook: assignment05-anushagali-info6105.ipynb

## Installation
To run these notebooks locally, you need to have the following installed:
- Python 3.x
- Jupyter Notebook
