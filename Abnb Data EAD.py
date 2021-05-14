#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests

from IPython.core.display import display, HTML

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time, os

import re


# In[2]:


import sys

import pickle
import random


# ### Listings

# In[3]:


with open('project/data/listing_master.pickle','rb') as read_file:
    listings = pickle.load(read_file)

listings.head()


# In[4]:


listings.info()


# In[5]:


listings_cleaned = listings.copy()


# In[6]:


# Clean up property types
def type_desc(prop_type):
    if "Entire place" in prop_type:
        return "Entire house"
    elif "Entire bungalow" in prop_type:
        return "Entire house"
    elif "Room in boutique hotel" in prop_type:
        return "Hotel room"
    elif "Tiny house" in prop_type:
        return "Entire house"
    elif "Entire villa" in prop_type:
        return "Entire house"
    elif "Aparthote" in prop_type:
        return "Entire serviced apartment"
    elif "Entire house" in prop_type:
        return "Entire house"
    elif "Entire condominium" in prop_type:
        return "Entire condominium"
    elif "Entire cabin" in prop_type:
        return "Entire cabin"
    elif "Private room" in prop_type:
        return "Private room"
    elif "Entire townhouse" in prop_type:
        return "Entire townhouse"
    elif "Entire apartment" in prop_type:
        return "Entire apartment"
    elif "Entire chalet" in prop_type:
        return "Entire chalet"
    elif "Entire guesthouse" in prop_type:
        return "Entire guesthouse"
    elif "Hotel room" in prop_type:
        return "Hotel room"
    elif "Entire loft" in prop_type:
        return "Entire loft"
    elif "Entire cottage" in prop_type:
        return "Entire cottage"
    elif "Entire guest suite" in prop_type:
        return "Entire guest suite"
    elif "Resort room" in prop_type:
        return "Resort room"
    elif "Camper/RV" in prop_type:
        return "Camper/RV"
    elif "Shared room" in prop_type:
        return "Shared room"
    elif "Entire serviced apartment" in prop_type:
        return "Entire serviced apartment"
    elif "Hostel beds" in prop_type:
        return "Hotel room"
    else:
        return prop_type


# In[7]:


listings_cleaned['type_desc'] = listings_cleaned['type'].map(lambda x: type_desc(x))


# In[8]:


listings_cleaned.type_desc.unique()


# In[9]:


# Clean up price
listings_cleaned['price_cleaned'] = listings_cleaned['price'].str.split(" ").apply(lambda x: x[0]).str.replace("$","").str.replace(",","").astype(float)


# In[10]:


listings_cleaned.drop(columns = 'price', inplace = True)
listings_cleaned.rename(columns = {"price_cleaned" : "price"}, inplace = True)


# In[11]:


# Clean up reviews
listings_cleaned['reviews_tmp'] = listings_cleaned['reviews'].str.replace("(", " ").str.replace(")","").str.split(" ")


# In[12]:


listings_cleaned['reviews_cleaned'] = listings_cleaned['reviews_tmp'].apply(lambda x: None if len(x) == 1 else x[0])
listings_cleaned['num_reviews'] = listings_cleaned['reviews_tmp'].apply(lambda x: x[1] if len(x) > 1 else None)


# In[13]:


listings_cleaned.head()


# In[14]:


listings_cleaned.drop(columns = ["reviews", "reviews_tmp"], inplace = True)
listings_cleaned.rename(columns = {'reviews_cleaned' :'reviews'}, inplace = True)


# In[15]:


listings_cleaned['reviews'] = listings_cleaned['reviews'].astype(float)
listings_cleaned['num_reviews'] = listings_cleaned['num_reviews'].astype(float)


# In[16]:


listings_cleaned.drop(columns = ["url"], inplace = True)
listings_cleaned.rename(columns = {"url_cleaned" : "url"}, inplace = True)
listings_cleaned.info()


# In[17]:


# Clean up hostel
listings_cleaned['accomdations_tmp'] = listings_cleaned['hostel'].str.split(" · ")

listings_cleaned['num_guets'] = listings_cleaned['accomdations_tmp'].apply(lambda x: x[0])
listings_cleaned['num_bedrooms'] = listings_cleaned['accomdations_tmp'].apply(lambda x: x[1] if len(x) >= 2 else None)
listings_cleaned['num_beds'] = listings_cleaned['accomdations_tmp'].apply(lambda x: x[1] if len(x) >= 3 else None)
listings_cleaned['num_baths'] = listings_cleaned['accomdations_tmp'].apply(lambda x: x[1] if len(x) >= 4 else None)


# In[18]:


listings_cleaned['num_guets'] = listings_cleaned['num_guets'].str.split(" ").apply(lambda x: x[0])
listings_cleaned['num_bedrooms'] = listings_cleaned['num_bedrooms'].str.split(" ").apply(lambda x: x[0] if x is not None else np.nan)
listings_cleaned['num_baths'] = listings_cleaned['num_baths'].str.split(" ").apply(lambda x: x[0] if x is not None else np.nan)
listings_cleaned['num_beds'] = listings_cleaned['num_beds'].str.split(" ").apply(lambda x: x[0] if x is not None else np.nan)


# In[19]:


#For studios, assign num_bedrooms = 0, num_baths = 1 and num_beds = 1 by default

listings_cleaned['num_bedrooms_cleaned'] = listings_cleaned['num_bedrooms'].apply(lambda x: "0" if x == 'Studio' else x)
listings_cleaned['num_baths_cleaned'] = listings_cleaned['num_baths'].apply(lambda x: "1" if x == 'Studio' else x)
listings_cleaned['num_beds_cleaned'] = listings_cleaned['num_beds'].apply(lambda x: "1" if x == 'Studio' else x)


# In[20]:


listings_cleaned.drop(columns = ["num_bedrooms", "num_baths", "num_beds"], inplace = True)


# In[21]:


listings_cleaned.rename(columns = {"num_bedrooms_cleaned" : "num_bedrooms",
                                   "num_baths_cleaned" : "num_baths",
                                   "num_beds_cleaned" : "num_beds"}, inplace = True)


# In[22]:


listings_cleaned['num_guets'] = listings_cleaned['num_guets'].astype(float)
listings_cleaned['num_bedrooms'] = listings_cleaned['num_bedrooms'].astype(float)
listings_cleaned['num_baths'] = listings_cleaned['num_baths'].astype(float)
listings_cleaned['num_beds'] = listings_cleaned['num_beds'].astype(float)


# In[23]:


columns = ['url', 'property_name', 'type_desc', 'price', 'reviews', 'num_reviews',
          'num_guets','num_bedrooms','num_baths', 'num_beds']


# In[24]:


listings_cleaned_final = listings_cleaned[columns]


# In[25]:


listings_cleaned_final.head()


# In[ ]:





# In[27]:


# Clean up listing details
with open('project/data/listing_details_final_all.pickle','rb') as read_file:
    listing_details = pickle.load(read_file)

listing_details.head()


# In[28]:


listing_details_cleaned = listing_details.copy()


# In[29]:


listing_details_cleaned['cleanliness'] = listing_details_cleaned['cleanliness'].astype(float)
listing_details_cleaned['accuracy'] = listing_details_cleaned['accuracy'].astype(float)
listing_details_cleaned['communication'] = listing_details_cleaned['communication'].astype(float)
listing_details_cleaned['location'] = listing_details_cleaned['location'].astype(float)
listing_details_cleaned['checkin'] = listing_details_cleaned['checkin'].astype(float)
listing_details_cleaned['value'] = listing_details_cleaned['value'].astype(float)


# In[30]:


listing_details_cleaned.dtypes


# In[31]:


listing_details_cleaned['response_rate_cleaned'] = listing_details_cleaned['response_rate'].str.split(":").apply(lambda x: x if x is None else x[1])
listing_details_cleaned['response_rate_cleaned'] = listing_details_cleaned['response_rate_cleaned'].str.replace("%","").astype(float)


# In[32]:


listing_details_cleaned.drop(columns = ['response_rate'], inplace = True)
listing_details_cleaned.rename(columns = {"response_rate_cleaned" : "response_rate"}, inplace = True)

listing_details_cleaned.head()


# In[33]:


listing_details_cleaned_final = listing_details_cleaned.copy()


# In[ ]:





# In[34]:


# Clean up location data
with open('project/data/location_all.pickle','rb') as read_file:
    locations = pickle.load(read_file)

locations.head()


# In[35]:


locations_cleaned = locations.copy()


# In[36]:


locations_cleaned['location_tmp'] = locations_cleaned['location'].str.split(",")


# In[37]:


locations_cleaned['region'] = locations_cleaned['location_tmp'].apply(lambda x: x[0])
locations_cleaned['state'] = locations_cleaned['location_tmp'].apply(lambda x: x[1])


# In[38]:


columns = ['url','region','state']

locations_cleaned_final = locations_cleaned[columns]


# In[39]:


locations_cleaned_final.head()


# In[40]:


#Merge Data


# In[41]:


tmp1 = listings_cleaned_final.merge(listing_details_cleaned_final,
                                   on = 'url',
                                   how = 'inner')


# In[42]:


tmp1.sort_values('url', inplace = True)


# In[43]:


tmp1.drop_duplicates('url',inplace = True)


# In[44]:


tmp1.info()


# In[45]:


listings_all_v1 = tmp1.merge(locations_cleaned_final,
                 on = 'url',
                 how = 'inner')


# In[46]:


listings_all_v1.sort_values('url', inplace = True)


# In[47]:


listings_all_v1.drop_duplicates('url',inplace = True)


# In[48]:


listings_all_v1.info()


# In[49]:


listings_all_v1.head()


# In[50]:


#Clean up amenities
with open('project/data/amenities.pickle','rb') as read_file:
    amenities = pickle.load(read_file)

amenities.head()


# In[51]:


amenities_cleaned = amenities.copy()


# In[52]:


amenities.vals.tolist()


# In[54]:


amenities_cleaned['vals_strings'] = [" ".join(map(str, item)) for item in amenities_cleaned['vals']]


# In[55]:


amenities_cleaned.head()


# In[56]:


amenities_cleaned['TV'] = [1 if 'TV' in x else 0 for x in amenities_cleaned['vals_strings']]
amenities_cleaned['Free_Parking'] = [1 if 'Free parking' in x else 0 for x in amenities_cleaned['vals_strings']]
amenities_cleaned['Wifi'] = [1 if 'Wifi' in x else 0 for x in amenities_cleaned['vals_strings']]
amenities_cleaned['Kitchen'] = [1 if 'Kitchen' in x else 0 for x in amenities_cleaned['vals_strings']]
amenities_cleaned['Heating'] = [1 if 'Heating' in x else 0 for x in amenities_cleaned['vals_strings']]
amenities_cleaned['Air_conditioning'] = [1 if 'Air conditioning' in x else 0 for x in amenities_cleaned['vals_strings']]


# In[57]:


amenities_cleaned.head()


# In[58]:


columns = ['url','TV','Free_Parking', 'Wifi', 'Kitchen', 'Heating', 'Air_conditioning']


# In[59]:


amenities_cleaned_final = amenities_cleaned[columns]


# In[60]:


listings_all_v2 = listings_all_v1.merge(amenities_cleaned_final,
                 on = 'url',
                 how = 'inner')


# In[61]:


listings_all_v2.sort_values('url', inplace = True)


# In[62]:


listings_all_v2.drop_duplicates('url',inplace = True)


# In[63]:


listings_all_v2.info()


# In[64]:


with open('project/data/listings_final.pickle', 'wb') as to_write:
    pickle.dump(listings_all_v2, to_write)


# In[ ]:




