import pandas as pd #data manipulation library
import quandl #dataset library quandl.com
import math

#df = quandl.get("FINRA/FORF_TLLTD", authtoken="xUdLbiaHu_9riN5AUeHR", start_date="1970-01-01", end_date="1970-01-01")
df = quandl.get('WIKI/GOOGL') #google stocks dataset
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']] #Valuable Features to work on 
#Adj. -> Adjusted are lower version of huge stocks for people to buy
#Open ->Opening balance of the day
#Close ->Closing Balance of the day
#High ->Highest price of stock of the day
#Low ->Lowest price of stock of the day
#Volume ->Volume of stock of the day

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0 #High level percentage change

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 #Percentage change of the day

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']] #Valuable features

print(len(df)) #length of dataset
#Our case output is 3424
forecast_col = 'Adj. Close' #Write the column whose value needs to be predicted

df.fillna(-99999, inplace=True) #Replace Nan by -9999 in dataset

forecast_out = int(math.ceil(0.001*len(df))) #example len(df) return 3082, and multiply it by 0.005 = 15.41 and ceil makes it 16 
#0.001*3424=ceil(3.424)=4 days
df['label'] = df[forecast_col].shift(-forecast_out) #To predict labelled col value by shifting forecast_col table by forecast_out
df.dropna(inplace=True) #optional for accuracy .To drop rows with Nan

print(df.head()) #Print first 5 rows
print(df.tail()) #Print last 5 rows
