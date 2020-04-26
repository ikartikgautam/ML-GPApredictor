import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

raw_data = pd.read_csv('./files/dummies.csv')

data = raw_data.copy()

data['Attendance'] = data['Attendance'].map({'Yes':1,'No':0})

print(data.describe())

y = data['GPA']
xr = data[['SAT','Attendance']]

x = sm.add_constant(xr)
results = sm.OLS(y,x).fit()

plt.scatter(data['SAT'],y)

yhat_yes = 0.6439 + (0.0014*data['SAT']) + (0.2226*1)
yhat_no = 0.6439 + (0.0014*data['SAT']) + (0.2226*0)
yhat = 0.275+(0.0017*data['SAT'])

plt.plot(data['SAT'],yhat_yes,c='green')
plt.plot(data['SAT'],yhat_no,c='red')
plt.plot(data['SAT'],yhat,c='blue')

plt.title('GPA of students predicted using their SAT scores')
plt.xlabel('SAT scores')
plt.ylabel('GPA scores')


#prediction

score = int(input('Enter your SAT score:'))
att = int(input('Is your attendance above 75%? 1=YES/0=NO'))

new_data = pd.DataFrame({'const':1,'SAT':[score],'Attendance':[att]})
#print(new_data)

prediction = results.predict(new_data)

pdf = pd.DataFrame({'Predicted GPA':prediction})
joined = new_data.join(pdf)

print('\n\nPREDICTED GPA VALUE:\n')
print(joined)

plt.show()