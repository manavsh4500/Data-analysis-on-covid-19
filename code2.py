"""
MANAV SHARMA
B20298
7668517941
"""

#importing libraries
import pandas as pd               
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.ar_model import AutoReg as AR

#2-------
#reading data
series = pd.read_csv('daily_covid_cases.csv', parse_dates=['Date'],
                      index_col=['Date'],
                      sep=',')
# Train test splitting
test_size = 0.35
X = series.values
#returns the smallest integer greater than or equal to a given number.
tst_sz = math.ceil(len(X)*test_size)
train = X[:len(X)-tst_sz]
test =X[len(X)-tst_sz:]

# Train autoregression
window = 5
model = AR(train, lags=window)
model_fit = model.fit()
coff=model_fit.params
print('Coefficients are : ' , coff)

#using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list() 

for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)] 
	yhat = coff[0] # Initialize to w0
	for d in range(window):
		yhat += coff[d+1] * lag[window-d-1] # Add other values 
	obs = test[t]
	predictions.append(yhat) #Append predictions to compute RMSE later
	history.append(obs) # Append actual test value to history, to be used in next step.

plt.figure()
plt.scatter(test, predictions)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Question 2-B(i)")
plt.show()

plt.figure()
plt.plot(test)
plt.plot(predictions, color='red')
plt.title("Q-2(B) -(ii)")
plt.legend(['Actual Data', 'Predicted data'])
plt.show()

# For RMSE :
n=len(test)
s=0
for i in range(n):
    s=s+(predictions[i]-test[i])**2
avg=sum(test)/len(test)
rmse=(math.sqrt(s/len(test))/avg)*100
print("RMSE is: " ,rmse)

# For MAPE :
s=0
for i in range(n):
    s=s+ abs(predictions[i]-test[i])/test[i]
mape=(s/n)*100
print("MAPE is: ",mape)

#3------


def auto_regression(p):
        Window = p # The lag=p
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

        ###  RMSE Calculation

        n=len(test)
        s=0
        for i in range(n):
            s=s+(predictions[i]-test[i])**2
        avg=sum(test)/len(test)
        rmse=(math.sqrt(s/len(test))/avg)*100

        ### MAPE Calculation

        s=0
        for i in range(n):
            s=s+ abs(predictions[i]-test[i])/test[i]
        mape=(s/n)*100

        return rmse[0],mape[0]
        
rmse=[0]*5
mape=[0]*5
#
rmse[0],mape[0]=auto_regression(1)
rmse[1],mape[1]=auto_regression(5)
rmse[2],mape[2]=auto_regression(10)
rmse[3],mape[3]=auto_regression(15)
rmse[4],mape[4]=auto_regression(25)

p=[1,5,10,15,25]
#
plt.figure()
plt.bar(p,rmse)
plt.xticks(p)
plt.xlabel("Lag-value")
plt.ylabel("RMSE(%)")
plt.show()
#
plt.figure()
plt.bar(p,mape)
plt.xticks(p)
plt.xlabel("Lag-value")
plt.ylabel("MAPE")
plt.show()

