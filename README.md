Task 1: Understanding Credit Risk
1. Introduction to Credit Risk
Credit risk refers to the possibility that a borrower will default on their debt obligations, which can include missing interest payments or failing to repay the principal. In financial institutions, assessing credit risk is crucial for determining whether to extend credit to individuals or businesses and under what terms. Credit scoring models help quantify this risk.

In this challenge, Bati Bank is seeking to implement a credit scoring model that classifies customers as high or low risk, using data provided by an eCommerce platform.

2. RFMS Formalism
The RFMS formalism (Recency, Frequency, Monetary value, Stability) is a popular approach for categorizing customers based on their transaction behavior. Here’s how it breaks down:

Recency: Measures how recently a customer made a transaction.
Frequency: Measures how often a customer makes transactions.
Monetary Value: Represents the value of the transactions a customer has made.
Stability: Indicates the variability of customer transactions over time.
In this project, RFMS formalism can be used as a proxy for creditworthiness. Customers with high recency, frequent transactions, and high monetary value may be considered lower risk, while customers with erratic transactions may be considered higher risk.

3. Weight of Evidence (WoE) and Information Value (IV)
Weight of Evidence (WoE) and Information Value (IV) are statistical methods used in credit scoring to transform categorical variables into continuous variables and assess their predictive power, respectively.

WoE: It measures the strength of a predictor in distinguishing between two categories, such as "good" (low risk) and "bad" (high risk) credit customers. Higher WoE values indicate stronger predictive power.
Information Value (IV): It quantifies the predictive power of a variable in relation to the target outcome. The IV can help in feature selection by identifying the most important predictors of credit default.
Why WoE and IV are Useful:
WoE helps deal with categorical variables, which are common in financial datasets (e.g., ProductCategory, ChannelId).
IV helps rank predictors based on their effectiveness in distinguishing between defaulters and non-defaulters.
Key IV thresholds:
IV < 0.02: Not useful for prediction
0.02 < IV < 0.1: Weak predictor
0.1 < IV < 0.3: Medium predictor
IV > 0.3: Strong predictor
Relevant Reference:
https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
Weight of Evidence and Information Value Explained
4. Credit Scoring Models
The primary goal of a credit scoring model is to estimate the probability of default (PD) for an individual or business. Models use past data to predict future default behavior.

Common approaches include:

Logistic Regression: A simple and widely used model for binary classification, which is useful for predicting whether a customer will default or not.
Decision Trees and Random Forests: Non-linear models that capture complex interactions between variables. These models may perform better with transactional data but can be more prone to overfitting.
Gradient Boosting Machines (GBM): A powerful ensemble technique that typically yields strong results in classification tasks.
In this challenge, we will likely experiment with multiple models (e.g., Logistic Regression, Random Forest) and evaluate them based on metrics like accuracy, precision, recall, and ROC-AUC.

Relevant Reference:
https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03
Developing a Credit Risk Model and Scorecard
5. Basel II Capital Accord and Default Definitions
The Basel II Capital Accord is a regulatory framework that provides guidelines for managing credit risk. It defines how banks should quantify their exposure to risk and maintain sufficient capital reserves to mitigate potential losses. In the context of this challenge, the definition of default may vary, but it generally refers to situations where a borrower fails to meet the terms of a loan or credit agreement, either by missing payments or failing to repay the principal.

Understanding Basel II is essential for aligning the credit scoring model with regulatory standards. This includes defining what constitutes a "bad" (high-risk) customer and ensuring the model complies with guidelines for risk measurement and management.

Relevant Reference:
https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf
Basel II Overview
6. Feature Selection for Credit Scoring
Selecting the right features is critical to building an effective credit scoring model. Based on the data provided, here are some key features we’ll likely use:

Amount: The value of a transaction. High transaction values may indicate creditworthiness, but frequent low-value transactions might signal different behaviors.
FraudResult: A binary feature indicating whether a transaction was fraudulent. This could be a strong predictor of risk.
ProductCategory, ChannelId: Categorical features indicating customer behavior patterns and preferences.
In feature engineering (Task 3), we’ll explore these variables and transform them using techniques like WoE, normalization, and aggregation.

7. Conclusion
Understanding credit risk and its components is crucial for building a robust credit scoring model. The RFMS framework and WoE/IV will play key roles in defining and selecting features. By applying models like logistic regression and decision trees, we will be able to predict default risk and assign credit scores to new customers, aligning the system with the business needs of Bati Bank.

