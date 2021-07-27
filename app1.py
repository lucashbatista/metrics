#importing packages 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
import pandas as pd
import datetime
import calendar
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#import quandl
import pymongo
from pymongo import MongoClient
#importing json
import json


#import data from csv file
df = pd.read_csv("data/incident.csv",header=0, encoding='unicode_escape')

#sort the following columns Opened > Created > Updated 
df.sort_values(by=['opened_at','sys_created_on', 'sys_updated_on'], ascending=True, inplace=True)

#transform column 'sys_created_on' to datatime using to_datetime
#datetime package should be imported
df['date'] = pd.to_datetime(df['sys_created_on']).dt.date
df['date'] = pd.to_datetime(df['date'])
#create new columns 
df['day_of_year']=df['date'].dt.dayofyear
df['week']=df['date'].dt.week
df['month']=df['date'].dt.month
df['day_of_week'] = df['date'].dt.day_name()
df['monthname'] = df['date'].dt.strftime("%B")
df['time']=df['date'].dt.time
df['year']=df['date'].dt.year
#make sure that all the NaN rows are removed from dataframe
df.dropna(how='all')
#Making a Connection with MongoClient
client = MongoClient("mongodb://localhost:27017/")
# database
db = client["tickets"]
# collection
metrics = db["metrics"]
#transform data into numpy arrays 
arr = df.to_numpy()


#create new datasets in order to work with MatPlotLib graphs 
caller=df.value_counts(subset=['caller_id'], normalize=False, sort=True, ascending=False).nlargest(20)
days=df.value_counts(subset=['day_of_week','monthname','year'], normalize=False, sort=True, ascending=False).nsmallest(30)
months=df.value_counts(subset=['monthname'], normalize=False, sort=True, ascending=False)
#labor_days = ['Monday','Tuesday','Wednesday','Thursday','Friday']
weekdays=df.value_counts(subset=['day_of_week'], normalize=False,ascending=True)
count_df=df.value_counts(subset=['sys_tags'], normalize=False, sort=True, ascending=False).nlargest()


#weekdays: number of tickets per day of week PIE Chart
barweekday = weekdays.plot(kind='bar',color='gray')
plt.title('IBMid Weekday Number of Tickets', fontdict=None, loc='center',fontweight='bold'
          ,size=12,color='black')
plt.xlabel('Weekday')
plt.ylabel('Number of Tickets')
plt.xticks(rotation=40)
plt.savefig('barweekday.png')

##number of tickets per weekdays and months combined.
pltdays=days.plot(kind='bar', stacked=True,color='blue')
plt.xticks(size=8,rotation=80)
plt.title('Number of Tickets Weekday',fontweight='bold',size=12)
plt.ylabel('Number of Tickets')
plt.savefig('pltdays.png')

#number of tickets per month
months.plot(kind='bar', stacked=True,color='blue')
plt.title('Monthly Tickets',fontweight='bold',size=12)
plt.xlabel('Months')
plt.ylabel('Tickets per Month')
plt.xticks(rotation=40)
plt.savefig('monthlytickets.png')




plthist=plt.hist(df['week'],bins=52,color='green')

#histogram of the number of tickets per week
plt.xlabel('Weekly',fontweight='bold')
plt.ylabel('Number of Tickets',fontweight='bold')
plt.title('IBMid Weekly Tickets', fontdict=None, loc='center',fontweight='bold',size=16,color='black', 
          pad=None)
plt.savefig('plthist.png')

#pie chart for Tags
pltpie=count_df.groupby(['sys_tags']).sum().plot(kind='pie', y='sys_tags',
                                                 counterclock=True,shadow=True)
plt.title("Pie Chart for IBMid Ticket Cases",fontname="Times New Roman",fontweight="bold",size=16)
plt.savefig('pltpie.png')

#number of tickets per problem/tag historigram graph
plotcount=count_df.plot(color='red')
plt.title('Number of Tickets per Case',fontweight='bold',size=12)
plt.xlabel('Tags',fontweight='bold')
plt.ylabel('Tickets per Tag',fontweight='bold')
plt.xticks(rotation=40)
plt.savefig('plotcount.png')

#number of tickets per problem/tag bar graph
countbar=count_df.plot(kind='bar', stacked=True,color='green')
plt.title('Number of Tickets per Case',fontweight='bold',size=12)
plt.xlabel('Tags',fontweight='bold')
plt.ylabel('Tickets per Tag',fontweight='bold')
plt.xticks(rotation=40)
plt.savefig('figure.countbar.png')

#remove columns for linear regression
df.drop(columns=['assignment_group', 'time_worked', 'caller_id',
       'short_description', 'sys_tags', 'assigned_to', 'state',
       'sys_created_by', 'opened_by', 'description', 'opened_at',
       'sys_created_on', 'sys_updated_on', ], inplace=True)

#groupby date and count number of tickets per day in order to work on the Linear Regression
day_df = df.groupby(['date']).count()

#training and testing data for tickets per day 

day_df.insert(0, 'enumerated', range(1, 1 + len(day_df)))

x = day_df[['number']]
y = day_df[['enumerated']]


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.05, random_state=0)
#create and fit the model

lm = LinearRegression().fit(X_train,Y_train)

#get intercept 

lm.intercept_

Y_pred = lm.predict(x)  # make predictions

#get coefficients 

[(col, coef) for col, coef in zip(X_train.columns,lm.coef_)]

#making predictions 

preds = lm.predict(X_test)
preds


#see the regression line 
plt.scatter(x, y)
plt.plot(x, Y_pred, color='red')
plt.show()
