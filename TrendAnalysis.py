# Adding all the imports
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib

# import json

# Initializing matplotlib parameters
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

path = 'data/StackOverflowTagsUpdated.csv'

data = pd.read_csv(path, parse_dates=['date'])
# Dataset is now stored in a Pandas Dataframe

# Print overview of the dataset
data.head()

data.describe()

# Set the tag which is to be analysed here
tag = data.loc[data['tag'] == 'java-7']

# Initialize the date variable and print the start date and end date from the dataset
tag['date'] = pd.to_datetime(tag['date'])
tag['date'].min(), tag['date'].max()

# Remove column Sparkline from the analysis
cols = ['sparkline']
tag.drop(cols, axis=1, inplace=True)
tag = tag.sort_values('date')
tag.isnull().sum()

# Group the tags by the date
tag = tag.groupby('date')['views'].sum().reset_index()

# Set the index of the tag as the date and show all indexes
tag = tag.set_index('date')
tag.index

# Resample the time series and assign to new variable y
y = tag['views'].resample('MS').sum()

# Get data for the year 2017
y['2017':]

# Plot y
y.plot(figsize=(15, 6))
plt.show()

# Applying ARIMA (Autoregressive Integrated Moving Average) model
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# Grid search to find the best parameters for the model
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

# For debugging only
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 1),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

# Test issue with Correlation before running this block
# results.plot_diagnostics(figsize=(16, 8))
# plt.show()

# Comparing real and predicted views for the tag within the year specified
pred = results.get_prediction(start=pd.to_datetime('2017-06-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2017':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Tag Views')
plt.legend()
plt.show()

# Only for debugging
# Finding the Mean Squared Error - measures the average of the squares of the errors (Average
# squared difference between the estimated and real values)
y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts are {}'.format(round(mse, 2)))

# Only for debugging
# Finding the Root Mean Squared Error - Difference between values predicted and the estimator of the real values
print('The Root Mean Squared Error of our forecasts are {}'.format(round(np.sqrt(mse), 2)))

# Predicting the forecast for the tag specified
pred_uc = results.get_forecast(steps=50)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='Observed Views', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast Views')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Tag Views')
plt.legend()
plt.show()

# Path for the bag of words
bagOfWords = 'data/my_bow.csv'

bow_data = pd.read_csv('my_bow.csv')
# Bag of Words is now stored in a Pandas Dataframe

# Print overview of the dataset
bow_data.head()

bow_data.describe()

# Filter dataset against bag of words and create separate dataframe
frame = data
trendList = []

for row in bow_data:
    trendRow = data.loc[data['tag'] == row]
    if trendRow is not None:
        trendList.append(trendRow)

final_df = pd.concat(trendList, ignore_index=True)
print(final_df)

# For debugging only
final_df['views_percentage_change'].max()

# Get the 50 biggest positive changes in views in the column views_percentage_change
trendingSkills = final_df.nlargest(50, 'views_percentage_change')

# Print the table with the 50 biggest positive changes in views in the column views_percentage_change
trendingSkills

# Top trending skills based on number of questions
# For debugging only
final_df['questions'].max()

# Get the 50 biggest positive changes in views in the column questions
trendingNoOfQuestions = final_df.nlargest(50, 'questions')

# Print the table with the 50 biggest positive changes in views in the column questions
trendingNoOfQuestions
