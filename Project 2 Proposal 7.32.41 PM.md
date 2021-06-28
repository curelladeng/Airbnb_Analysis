### Project 2 Proposal

#### Analysis Summary:
The goal of this project was to use linear regression model to identify features that are highly correlated with rental price on Airbnb, in order to help Airbnb hosts understand which property features could help to ask for higher rental price. The data I worked with contains rental price as the target variable, and some geographic and categorical attributes of active listings in Lake Tahoe. The final linear regression model was applied Lasso Regularization to help with feature selections, and RMSE was used to evaluate the model performance. This model was able to identify that #accomdates, #bathrooms and the location scores are 3 major features to impact rental price.

<img src="feature importance.png" width=500>

#### Data Description:
The unit of analysis will be active listings in South Lake Tahoe.
The data features that will be included in the modelings are:
* listing_id
* property_type
* capacity
* number_bedroom
* number_bathroom
* number_bed
* number_reviews
* review_score
* host_status

The target variable is the listing_price

#### Tools:
* Python3 BeautifulSoup + Selenium
