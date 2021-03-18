#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 00:04:40 2020

@author: dhruvpanchal
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline #only works with Jupyter notebook or Ipython
#plt.ion() #to generate plots in line as we code plot 
import numpy as np
# Set option to display all the columns
pd.set_option('display.max_columns', None)
#read the dataset file and assign dataframe
df = pd.read_csv('fake_job_postings.csv')

#print the dataframe for top and bottom 5 rows and to get total rows and columns
print(df.head(-5))
print(df.shape)
#Print the unique values in each column
print(df.nunique())
# identify null values in each column
print(df.isna().sum())
#Calculate null values per total entries
print(df.isna().sum()/len(df))

df2= df.copy()

#Salary_range, department, benefits, and job_id adds no value and has a lot of missing value, hence removing those
df2.drop(['salary_range', 'job_id', 'department', 'benefits'], axis = 1, inplace = True)

#check number of rows and columns for new data frame
print(df2.shape)
#sort the values in title column and set the index back to normal
df2 = df2.sort_values('title').reset_index(drop = True)

#replacing the null values by backward filling

df2['employment_type'] = df2['employment_type'].bfill(axis=0)
df2['required_experience'] = df2['required_experience'].bfill(axis=0)
df2['required_education'] = df2['required_education'].bfill(axis = 0)
df2['industry'] = df2['industry'].bfill(axis=0)
df2['function'] = df2['function'].bfill(axis=0)

df3= df2.copy()
#removing the row with empty description
df3 = df3[df3['description'].notna()]
# null values in the data frame

print(df3.isna().sum())
#Imputing all the rows that has any missing values in it

df3 = df3.dropna(axis = 0, how = 'any')
#Imputing duplicated rows created after bfill
df3 = df3.drop_duplicates(keep = 'first')

df4 = df3.copy()

#Combining company profile, decription and requirements in one column

df4['description'] = df4['description'] + ' ' + df4['requirements'] + ' ' + df4['company_profile']
df4.drop(['company_profile', 'requirements'], axis = 1, inplace = True)
#Spliting the location column based on country code and city
df4['country_code'] = df4['location'].str.split(',', expand=True)[0]
df4['city'] = df4['location'].str.split(',', expand = True)[2]

#Identifying all the null values in the city column
df4.loc[df4['city'] == ' ', 'city'] = np.nan
#identify remaining null values 
print(df4.isna().sum())
#removing those 992 null rows with empty cities in it
df4.dropna(inplace = True)

#for building a list of countries
import pycountry 
#Accessing alpha_2 for the country list and checking it for the country code in the dataset
list_alpha_2 = [i.alpha_2 for i in list(pycountry.countries)]
#defnining a function to access country from list based on its code
def country(df):
    if df['country_code'] in list_alpha_2:
        return pycountry.countries.get(alpha_2 = df['country_code']).name
    
#Adding new column for the country name   
df4['country_name'] = df4.apply(country, axis = 1)
#removing redundant location and country code column
df4.drop(['location', 'country_code'], axis = 1, inplace = True)

from wordcloud import WordCloud, STOPWORDS
plt.figure(figsize = (20,20))
#using exisitng stop words
stopwords = set(STOPWORDS)
#plotting the wc for non fraudulent jobs based on the text in description
wc = WordCloud(background_color = "white", stopwords=stopwords, width = 1600 , height = 800 , max_words = 3000).generate(" ".join(df4[df4.fraudulent == 0]['description']))
#removing the axis to just display the whole word cloud plot
plt.axis("off")
plt.imshow(wc , interpolation = 'sinc')
plt.savefig('genuine_cloud.jpeg')
# making a word cloud plot for fruadulent job posting
plt.figure(figsize = (20,20))
stopwords = set(STOPWORDS)
wc = WordCloud(background_color="black", stopwords=stopwords, width = 1600 , height = 800 , max_words = 3000).generate(" ".join(df4[df4.fraudulent == 1]['description']))

plt.axis("off")
plt.imshow(wc , interpolation = 'bilinear')
plt.savefig('fraud_cloud.jpeg')

#saving the clean dataset
df_clean = df4.copy()
df_clean.to_csv('clean Job posting.csv') 

#Checking categorical columns and ploting them
df4.nunique()

categorical_columns = []
for col in df4.columns:
    print('Identified unique values in {0} are:'.format(col), df4[col].nunique())
    if df4[col].nunique() < 40:
        categorical_columns.append(col)
print('Categorical columns:',categorical_columns)
#Plotting categorical column and checking the distribution
#Using subplots to plot all together in a single layout

#Defining plot rows and columns and individual figure size
fig, axs = plt.subplots(len(categorical_columns)//2 , 2, figsize = (35,40))
plt_row = 0
plt_col = 0

for i, col in enumerate(categorical_columns):
#skipping fraudulent column for categorical classification
    if col == 'fraudulent':
        continue
#Counting unique values in each categorical column    
    a = df4[col].value_counts()
#Ploting label or index on x axis and its count on y axis.
    sns.barplot(x= a.index, y= a, ax = axs[plt_row][plt_col])
    axs[plt_row][plt_col].set_title('Distribution of {0}'.format(col), size = 20)
#Acessing labels and rotating to 90 degrees for better viewability  
    for label in axs[plt_row,plt_col].get_xticklabels():
        label.set_rotation(90)
    if plt_col == 0:
        plt_col = 1
    else:
        plt_col = 0
        plt_row += 1
plt.show()

#Ploting a last plot of the target variable as the fraudulent column
last_plt = plt.figure(figsize = (20,10))
fraud_distribution = df4['fraudulent'].value_counts()
sns.barplot(x= fraud_distribution.index, y= fraud_distribution)
plt.title('Distribution of target variable', size = 20)
plt.show()








