"""
MANAV SHARMA
B20298
7668517941
"""
#importing libraries
import pandas as pd           
import math
from scipy.stats import pearsonr
from statsmodels.tsa.ar_model import AutoReg as AR

#reading data
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = list(series['new_cases'])
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]
n=len(train)
p=1
flag=1
while(flag==1):
    new_train=train[p:]
    l=len(new_train)
    lag_new_train=train[:n-p]
    corr, _ = pearsonr(lag_new_train,new_train)
    if(2/math.sqrt(l)>abs(corr)):
        flag=0
    else:
        p=p+1

print(p-1)
Window = p-1 
model = AR(train, lags=Window) 
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
history = train[len(train)-Window:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-Window,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(Window):
        yhat += coef[d+1] * lag[Window-d-1] # Add other values
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.

n=len(test)
s=0
for i in range(n):
    s=s+(predictions[i]-test[i])**2
avg=sum(test)/len(test)
rmse=(math.sqrt(s/len(test))/avg)*100
print("RMSE IS: ",rmse)

# For MAPE :

s=0
for i in range(n):
    s=s+ abs(predictions[i]-test[i])/test[i]
mape=(s/n)*100
print("MAPE IS: ",mape)