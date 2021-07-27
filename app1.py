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



df = pd.read_csv("data/incident.csv",header=0, encoding='unicode_escape')




