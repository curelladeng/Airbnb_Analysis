# Airbnb Rental Price Predictor
Wenting Deng

## Abstract
The goal of this project was to use linear regression models to identify features that are highly correlated with rental price on Airbnb, in order to help Airbnb hosts understand which property features could help to ask for higher rental price. The data I worked with were scrapped from [Airbnb](www.airbnb.com). The dataset contains rental price as the target variable, and some geographic and categorical attributes of active listings in Lake Tahoe. After finalizing the model I was able to find features that impact rental price most in Lake Tahoe.

## Design
Airbnb is a home-sharing platform that allows home hosts to put their properties online, and it's mainly for short-term rent. Airbnb hosts are expected to set their own prices for listings. Although Airbnb and some other online resources provide guidance in pricing optimization, it's always a challenge for hosts to understand what could impact the price. In this case, the linear regression model would be useful to understand feature importance regarding rental price, which could enable hosts understand what factors can they try to improve in order to ask for better price.

## Data
The data that scrapped from [Airbnb](www.airbnb.com) contains 1,261 unique listings and 27 features. The features include property locations, property types, amenities, host levels(whether a superhost) as categorical variables. And the numerical variables include number of reviews, rating scores of each listing, number of bedrooms/bathrooms, accommodates etc as numerical variables. The target variable is the rental price. Given the large number of categories in property type and location, they two features were further grouped into more general categories. After data EAD, 847 observations left.


## Algorithms
**Features Engineering**
1. Map various property type into 3 major categories: House, Condominium and others
2. Map location into neighborhoods and further group them into regions by the lake and regions further to the lake
3. Convert categorical variables into dummies
4. Combine rating scores to calculate average ratings to reduce feature collinearity


**Model Evaluation and Selection**
The entire dataset of 847 observations were randomly split into training, validation and testing sets by 60/20/20. The 5-fold cross validation were leveraged on training and validation sets for model evaluation and selection. And the selected model was applied to testing data for prediction.

**5-fold cross validation on 18 features**

| Model| Ajusted R^2 |
| --- | --- |
| Linear Regression | 0.54 |
| Lasso | 0.55 |

Lasso regression was selected as my final model.

**Testing on Lasso Regression**
R^2: 0.59
RMSE: 91.69

**Linear Regression Assumptions Testing**
<img src="model plot" width=500>

<img src="Q-Q plot" width=500>

<img src="res dist" width=500>

**Feature Importance**

<img src="feature importance" width=500>

**Tools**
* BeautifulSoup and Selenium for web scraping
* Numpy and Pandas for data manipulation
* Scikit-learn & StatsModels for model evaluation and selection
* Matplotlib and Seaborn for plotting

## Communication
In addition to the slides and visuals presented, it will be embedded on my analytic blog.
