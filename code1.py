"""
MANAV SHARMA
B20298
7668517941
"""

#IMPORTING LIBRARIES
import pandas as pd               
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.graphics.tsaplots as sm

#READING DATA
df=pd.read_csv("daily_covid_cases.csv")
#LIST WITH NAME OF COL AS ITEMS
colb=df.columns

#LIST OF PARTICULAR COLUMN
dl=list(df[colb[0]])
av=list(df[colb[1]])

tot=len(dl)
values=['Feb-20','Apr-20','Jun-20','Aug-20','Oct-20','Dec-20','Feb-21','Apr-21','Jun-21','Aug-21','Oct-21']
x=[]
for i in range(tot):
    x.append(i)
xlen=[x for x in range(int(tot/11),tot,int(tot/11))]

plt.figure()
plt.plot(x,av)
plt.xticks(xlen,values,rotation=90)
plt.xlabel('Month-Year')
plt.ylabel('Daily confirmed cases')
plt.show()

#list with all elements of av expect last one i.e n-1 th one
lag=av[:len(av)-1]
#list with all elements of av expect first one i.e 0th one
new_av=av[1:]

corr, _ = pearsonr(lag,new_av)
print("VALUE OF CORRELATION IS: ",corr)

plt.figure()
plt.scatter(new_av,lag)
plt.xlabel('x(t)')
plt.ylabel('x(t-1)')
plt.xticks([x for x in range(0,int(max(av)),50000)],[x for x in range(0,400000,50000)],rotation=90)
plt.show()


n=len(av)
Lag_Series=[]
for i in range(1,7):
    Lag_Series.append(av[:n-i])
pearson_correlation=[]
for i in range(1,7):
    corr, _ = pearsonr(Lag_Series[i-1],av[i:])
    pearson_correlation.append(corr)
plt.figure()
plt.plot([x for x in range(1,7)],pearson_correlation)
plt.ylim(0.9500,1)
plt.ylabel('Correlation Coefficient')
plt.xlabel('Lag-value')
plt.show()


#This creates one graph with the scatterplot of observed values compared to fitted values.
sm.plot_acf(av,lags=6)
plt.show()

