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


# ### Scrape Listing Pages By Using BeautifulSoup

# In[ ]:


# Create url list to contain all listing pages
url_list = []

for page in range(20, 281, 20):
    url = "https://www.airbnb.com/s/South-Lake-Tahoe--CA--United-States/homes?tab_id=home_tab&refinement_paths%5B%5D=%2Fhomes&flexible_trip_dates%5B%5D=june&flexible_trip_dates%5B%5D=may&flexible_trip_lengths%5B%5D=weekend_trip&date_picker_type=calendar&checkin=2021-06-01&checkout=2021-06-06&source=structured_search_input_header&search_type=pagination&place_id=ChIJ4U-l7omFmYARjpEj-0YvG90&federated_search_session_id=83ffa40c-4883-4160-97bb-e1f3fdb22761&items_offset="+str(page) + "&section_offset=3"
    url_list.append(url)


# In[ ]:


# Insert the first search page because the url is very different
url_first = "https://www.airbnb.com/s/South-Lake-Tahoe--CA--United-States/homes?tab_id=home_tab&refinement_paths%5B%5D=%2Fhomes&flexible_trip_dates%5B%5D=june&flexible_trip_dates%5B%5D=may&flexible_trip_lengths%5B%5D=weekend_trip&date_picker_type=calendar&checkin=2021-06-01&checkout=2021-06-06&source=structured_search_input_header&search_type=autocomplete_click&query=South%20Lake%20Tahoe%2C%20CA%2C%20United%20States"
url_list.insert(0, url_first)


# In[ ]:


len(url_list)


# In[ ]:


user_agent = {'User-Agent': 'Mozilla/5.0'}

property_name, listing_url, property_type, price = [],[],[],[]
reviews = []
hostel = []

for url in url_list:
    response = requests.get(url, headers = user_agent)
    soup = BeautifulSoup(response.text, "html.parser")
    
        
    for item in soup.select('[itemprop = itemListElement]'):
        property_name.append(item.select('meta')[0]['content'])
        listing_url.append(item.select('meta')[2]['content'])
        property_type.append(item.find_all('div', class_='_b14dlit')[0].text)
        price.append(item.find_all('span', class_='_krjbj')[0].text)
        
        if len(item.find_all('span', class_='_18khxk1'))>0:
            reviews.append(item.find_all('span', class_='_18khxk1')[0].text)
        else:
            reviews.append("")
            
        if len(item.find_all('div', class_='_kqh46o')) > 0:
            hostel.append(item.find_all('div', class_='_kqh46o')[0].text)
        else:
            hostel.append("")
        


# In[ ]:


# Save raw data into a dictionary
keys = ['url','property_name','type', 'price', 'reviews', 'hostel']
listing_dic = dict(zip(keys, [listing_url,
                              property_name,
                              property_type,
                              price,
                              reviews,
                              hostel]))


# In[ ]:


listing_df = pd.DataFrame(listing_dic)


# In[ ]:


import sys

import pickle
import random


# In[ ]:


# Save into pickle for future analysis
with open('listings_final.pickle', 'wb') as to_write:
    pickle.dump(listing_df, to_write)


# In[ ]:





# ### Scrape Listing Details By Using BeautifulSoup And Selenium

# In[2]:


chromedriver = "/Applications/chromedriver" # path to the chromedriver executable
os.environ["webdriver.chrome.driver"] = chromedriver


# In[4]:


with open('project/listing_master.pickle','rb') as read_file:
    listing_master = pickle.load(read_file)

listing_master.head()


# In[5]:


listing_master.info()


# In[6]:


url_list = listing_master['url_cleaned'].tolist()


# In[7]:


len(url_list)


# In[8]:


cleanliness = []
accuracy = []
communication = []
location = []
checkin = []
value = []
urls = []
response_rate = []
host_status = []
identify_verify = []
neighboorhood = []


# In[331]:


driver = webdriver.Chrome(chromedriver)


# In[ ]:


for url in url_list:
    urls.append(url)
    
    driver.get(url)
    time.sleep(5+np.random.rand())
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    try:
        if len(soup.find_all('span', class_ = '_4oybiu')) > 0:
            cleanliness.append(soup.find_all('span', class_ = '_4oybiu')[0].text)
            accuracy.append(soup.find_all('span', class_ = '_4oybiu')[1].text)
            communication.append(soup.find_all('span', class_ = '_4oybiu')[2].text)
            location.append(soup.find_all('span', class_ = '_4oybiu')[3].text)
            checkin.append(soup.find_all('span', class_ = '_4oybiu')[4].text)
            value.append(soup.find_all('span', class_ = '_4oybiu')[5].text)
        else:
            cleanliness.append(None)
            accuracy.append(None)
            communication.append(None)
            location.append(None)
            checkin.append(None)
            value.append(None)

            
        if len(soup.find_all('li', class_ = '_1q2lt74', text=re.compile(r'Response rate'))) > 0:
            response_rate.append(soup.find_all('li', class_ = '_1q2lt74', text=re.compile(r'Response rate'))[0].text)
        else:
            response_rate.append(None)
            
        if len(soup.find_all('span', class_ = '_pog3hg', text = re.compile('Superhost'))) > 0:
            host_status.append(soup.find_all('span', class_ = '_pog3hg', text = re.compile('Superhost'))[0].text)
        else:
            host_status.append("Host")
            
        if len(soup.find_all('span', class_ = '_pog3hg', text = re.compile('Identity verified'))) >0:
            identify_verify.append(soup.find_all('span', class_ = '_pog3hg', text = re.compile('Identity verified'))[0].text)
        else:
            identify_verify.append("Not Verified")
       

    except Exception as e:
        time.sleep(5+np.random.rand())
        pass 


# In[15]:


listings_keys = ['url', 'cleanliness', 'accuracy', 'communication', 'location',
                'checkin', 'value', 'response_rate', 'identify_verify','host_status']        
listings_dic = dict(zip(listings_keys, [urls,
                                        cleanliness,
                                        accuracy,
                                        communication,
                                        location,
                                        checkin,
                                        value,
                                        response_rate,
                                        identify_verify,
                                        host_status]))  


# In[16]:


listing_details = pd.DataFrame(listings_dic)


# In[20]:


with open('listing_details_final.pickle', 'wb') as to_write:
    pickle.dump(listing_details, to_write)


# In[158]:


driver.quit()


# In[341]:


driver = webdriver.Chrome(chromedriver)


# In[188]:


#Neighborhood
urls,location = [],[]

for url in url_list:
    urls.append(url)
    
    driver.get(url)
    time.sleep(5+np.random.rand())
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    try:
        if len(soup.find_all('span', class_ = '_169len4r')[0]) > 0:
            location.append(soup.find_all('span', class_ = '_169len4r')[0].text)
        else:
            location.append(None)
    
    except Exception as e:
        time.sleep(8+np.random.rand())
        pass 

   


# In[190]:


listings_keys = ['url', 'location']        
listings_dic = dict(zip(listings_keys, [urls,
                                        location])) 
    


# In[191]:


location_all = pd.DataFrame(listings_dic)


# In[197]:


with open('location_all.pickle', 'wb') as to_write:
    pickle.dump(location_all, to_write)
    


# In[198]:


driver.quit()


# In[ ]:


#Amenities


# In[243]:


driver = webdriver.Chrome(chromedriver)


# In[306]:


amenities_dict = dict()


# In[307]:


for url in add_url:
    driver.get(url)
    time.sleep(6+np.random.randn())
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    lst = []
    try: 
        for div in soup.find_all('div', class_ = 'iikjzje dir dir-ltr'):
            for item in div.find_all('div', class_=""):
                lst.append(item.text)
                amenities_dict[url] = lst
    except Exception as e:
        time.sleep(8+np.random.randn())
        pass 


# In[308]:


amenities_dict


# In[310]:


amenities_keys = amenities_dict.keys()


# In[311]:


df_amenities = pd.DataFrame(amenities_keys, columns = ["url"])


# In[312]:


amenities_vals = amenities_dict.values()


# In[313]:


df_amenities['vals'] = amenities_vals


# In[329]:


with open('amenities.pickle', 'wb') as to_write:
    pickle.dump(df_amenities, to_write)


# In[366]:


driver.quit()

