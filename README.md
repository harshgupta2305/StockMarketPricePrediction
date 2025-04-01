# StockMarketPricePrediction
!pip install chart_studio
import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

#for offline plotting
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot #Fixed the typo here
init_notebook_mode(connected=True)

tesla = pd.read_csv('/content/TSLA (1).csv')
tesla.head()

tesla.info()

tesla['Date']  = pd.to_datetime(tesla['Date'])

print(f'Dataframe contain stock price between {tesla.Date.min()} {tesla.Date.max()}') # Changed tesla.Data to tesla.Date
print(f'Total days = {(tesla.Date.max() - tesla.Date.min()).days} days') # Changed tesla.Data to tesla.Date

tesla.describe()

tesla[['Open','High','Low','Adj Close']].plot(kind='box')

#regresssion model applying
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

#split tesla stock data into training and testing
X = np.array(tesla.index).reshape(-1,1)
Y = tesla['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=101)

#feature scaling
scaler = StandardScaler().fit(X_train)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)

trace0 = go.Scatter(
    x = X_train.T[0],
    y = Y_train,
    mode = 'markers',
    name = 'Actual'
)
trace1 = go.Scatter(
    x = X_train.T[0],
    y = lm.predict(X_train).T,
    mode = 'lines',
    name = 'Predicted'
)
tesla_data = [trace0,trace1]
layout.xaxis.title.text = 'Day'
plot2 = go.Figure(data=tesla_data, layout=layout)

iplot(plot2)

#calculate scores for model evaluation
score = f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train, lm.predict(X_train))}\t{r2_score(Y_test, lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train, lm.predict(X_train))}\t{mse(Y_test, lm.predict(X_test))}
'''
print(score)
