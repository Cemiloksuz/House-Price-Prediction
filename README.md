# House-Price-prediction

Description

What is DataScience_House-Price-Prediction?
Different people have different needs when they want to buy a house. How many rooms, baths, how far it is from the city center, how easy it is to get there, etc. How do all of these factors work together to figure out how much a house costs? The goal of this project is to make a model that can predict home prices in Ames, Iowa, using 79 different factors. Data analysis and variable engineering were used in the project to get the data ready for the machine learning model. After that, several machine learning models were tried out, and the best one was chosen.

Article Content
There are two different.csv files, train and test. This is because the dataset comes from a Kaggle challenge. The test sample doesn't have any house prices, so you have to guess what they are.

Columns
You can use this link to examine the data. https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation

Flow
The data was uploaded and examined.
Columns are categorical and numerical. Both hierarchical and non-hierarchical columns were recognized for encoding.
Analyses included categorical and numerical variables.
Target variable-based categorical variables were evaluated.
We analyzed outliers. Suppressing outliers created upper and lower bounds.
Missing column values were recorded. According to missing value rate, several solutions were used.
Hierarchical categorical columns were encoded.
Rare value analysis merged underrepresented categorical variables.
Additional columns were created using feature engineering.
Label endocing was used on binary variables.
Non-hierarchical and multi-class categorical variables were one-hot encoded.
Numerical variables were standardized. Logarithmic transformation was applied to the target variable.
Test and train data were separated at research start.
Over train data, dependent and independent variables were constructed.
Model selection algorithms were used to build models.
The model was created and hyperparameter optimized using Light GBM.
We examined feature significance.
Finally, prediction data was created.
