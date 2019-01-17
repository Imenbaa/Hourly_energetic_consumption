import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb


#################### Load Dataset from excel file
dataset=pd.read_excel('hourly_cosumption_test.xls')
dataset ['Main Load [kWh]']=dataset['Main Load [kWh]'].astype(float)

#################### Transform to timeseries
index = pd.DatetimeIndex(dataset['Date'])
dftime= dataset.rename(lambda x: dataset['Date'][x].strftime('%Y-%m-%d %H:%M:%S'))
consumption=dftime['Main Load [kWh]'].fillna(0)

dataset = pd.Series(consumption,index=index)
##################### plot our data
color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
_ = dataset.plot(style='.', figsize=(7,5), color=color_pal[2], title='Energy Consumption')

##################### Sum weekly consumption
consumptionbyweek = dataset.groupby([lambda x: x.year ,lambda x: x.weekofyear])
sumbyweek = consumptionbyweek.sum() 

sumbyweek.unstack(0).plot(kind='line',figsize=(15,6), title='Weekly Energy Comsumption')
plt.xlabel('Week of year')
plt.ylabel('Total consumption')
print(sumbyweek)
##################### Min and Max 
minmax = dataset.groupby([lambda x: x.year ,lambda x: x.weekofyear])
############### Min weekly
minbyweek = pd.concat([minmax.idxmin(),minmax.min()],axis=1,ignore_index=True)
minbyweek = minbyweek.rename({0:'Date of Min' , 1:'Min Main Load [kWh]'},axis=1)
s = []
s2 = []
for line in minbyweek['Date of Min']:
    s.append(line.strftime('%Y-%m-%d'))
for line in minbyweek['Min Main Load [kWh]']:
    s2.append(line)
    
minweek = pd.Series(s2)
minweek=minweek.rename(pd.Series(s))
##########Visualize the comparaison of years
minweek.plot(kind='bar')
plt.title('Minimun Weekly Consumption')
plt.xlabel('Week')
plt.ylabel('Minimun consumption')

##############Max weekly
#maxbyweek = dataset.loc[minmax.idxmax()]
#maxbyweek=maxbyweek.rename(lambda x:x.strftime('%Y-%m-%d')) 
maxbyweek = pd.concat([minmax.idxmax(),minmax.max()],axis=1,ignore_index=True)
maxbyweek = maxbyweek.rename({0:'Date of Max' , 1:'Max Main Load [kWh]'},axis=1)

s1 = []
s3 = []
for line in maxbyweek['Date of Max']:
    s1.append(line.strftime('%Y-%m-%d'))
for line in maxbyweek['Max Main Load [kWh]']:
    s3.append(line)
    
maxweek = pd.Series(s3)
maxweek=maxweek.rename(pd.Series(s1))


maxweek.plot(kind='bar') #unstack(0).
plt.title('Maximum Weekly Consumption')
plt.xlabel('Week of year')
plt.ylabel('Maximum consumption')
############### weekly max min
weeklyminmax=pd.concat([minbyweek,maxbyweek],axis=1,ignore_index=True)
weeklyminmax=weeklyminmax.rename({0:'Date of Min' , 1:'Min Main Load [kWh]',
                     2:'Date of Max' , 3:'Max Main Load [kWh]'},axis=1)
print(weeklyminmax)

############### Comparison between years consumption
consumptionbyyear = dataset.groupby([lambda x: x.year])
sumbyyear = consumptionbyyear.mean()
sumbyyear.plot(kind='bar')
plt.title('Mean Consumption by year')
plt.xlabel('Year')
plt.ylabel('Consumption')

############### Forecasting #############################################################################

def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X
#######Visualize my features
X, y = create_features(pd.DataFrame(dataset), label='Main Load [kWh]')
features_and_target = pd.concat([X, y], axis=1)
sns.pairplot(features_and_target.dropna(),
             hue='hour',
             x_vars=['hour','dayofweek','year','weekofyear'],
             y_vars='Main Load [kWh]',
             height=5,
             plot_kws={'alpha':0.1, 'linewidth':0}
            )
plt.suptitle('Energy consumed by Hour, Day of Week, Year and Week of Year')
plt.show()
########Now i will choose my forecasting horizon
split_date = '01-10-2017'
dataset_train = pd.DataFrame(dataset.loc[dataset.index <= split_date].copy())
dataset_test = pd.DataFrame(dataset.loc[dataset.index > split_date].copy())
##########Visualize my Train and Test data
_ = dataset_test.rename(columns={'Main Load [kWh]': 'TEST SET'}) \
    .join(dataset_train.rename(columns={'Main Load [kWh]': 'TRAINING SET'}), how='outer') \
    .plot(figsize=(10,5), title='Energy Consumption', style='.')

###########Split my data from labels and assign them to variables
X_train, y_train = create_features(dataset_train), dataset_train['Main Load [kWh]']
X_test, y_test   = create_features(dataset_test), dataset_test['Main Load [kWh]']
#########Build The Model
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=100, #stop if 100 consequent rounds without decrease of error
        verbose=False)
###########Importance of features 
xgb.plot_importance(reg, height=0.9)

#######Visualize result of prediction
def plot_performance(base_data, date_from, date_to, title=None):
    plt.figure(figsize=(15,3))
    if title == None:
        plt.title('From {0} To {1}'.format(date_from, date_to))
    else:
        plt.title(title)
    plt.xlabel('time')
    plt.ylabel('energy consumed')
    plt.plot(dataset.index,dataset, label='data')
    plt.plot(X_test.index,X_test_pred, label='prediction')
    plt.legend()
    plt.xlim(left=date_from, right=date_to)
X_test_pred = reg.predict(X_test)  
plot_performance(dataset,dataset.index[0].date(), dataset.index[-1].date(),
                 'Original and Predicted Data')
plot_performance(y_test, y_test.index[0].date(), y_test.index[-1].date(),
                 'Test and Predicted Data')
plot_performance(y_test, '01-01-2018', '31-01-2018', 'January 2018 Snapshot')
plt.legend()
plt.show()
#############visualize forcast for five random weeks
random_weeks = X_test[['year', 'weekofyear']].sample(5)
for week in random_weeks.iterrows():
    index = (X_test.year == week[1].year) & \
            (X_test.weekofyear == week[1].weekofyear)
    data = y_test[index]
    plot_performance(data, data.index[0].date(), data.index[-1].date())
######Error Mesure
mean_absolute_error(y_true=y_test,
                   y_pred=X_test_pred)
###############Percent of Mean Error for five Random Weeks
error_by_week = []
random_weeks = X_test[['year', 'weekofyear']].sample(5)
for week in random_weeks.iterrows():
    index = (X_test.year == week[1].year) & \
            (X_test.weekofyear == week[1].weekofyear)
    error_by_week.append(mean_absolute_error(y_test[index], X_test_pred[index]))
pd.Series(error_by_week, index=random_weeks.index)