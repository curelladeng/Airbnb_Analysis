### Project 2 Proposal

#### Analysis Objective:
In this project, I'm going to use the Airbnb listing data in South Lake Tahoe to predict Airbnb rental price by using multiple linear regression.
This model will help Airbnb hosts to identify attributes that are highly relevant to rental price.


#### Data Description:
The data will be collected via web scraping from [Airbnb](https://www.airbnb.com/)
And the unit of analysis will be active listings in South Lake Tahoe.
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

#### MVP Goal:
* Find variables that are highly correlated to listing price.
