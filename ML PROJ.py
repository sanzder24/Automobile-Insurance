import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from math import sqrt

import numpy as np
import gc


from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler


dataFrame = pd.read_csv('car_insurance_claim_2.csv', delimiter = ',')
sns.set(style = 'white', context = 'notebook', palette = 'deep')


def f(row):
    if row['CLM_AMT'] > 0 :
        val = 1
    else:
        val = 0.
    return val

#  ####################Data Cleanup###############################
def cleanup_data(data_frame):
	global dataFrame
	# dataFrame[['Day', 'MonthYr']] = dataFrame['Day'].str.split(n = 2, expand = True)
	dataFrame['DAY'] = dataFrame['BIRTH'].str[0:2]
	dataFrame['MONTH'] = dataFrame['BIRTH'].str[3:6]
	dataFrame['YEAR'] = '19' + dataFrame['BIRTH'].str[7:9]
	dataFrame.DAY = dataFrame.DAY.astype(int)
	dataFrame.YEAR = dataFrame.YEAR.astype(int)
	data_frame.drop(['BIRTH'], axis = 1, inplace = True)
	dataFrame.loc[(pd.isnull(dataFrame.AGE)), 'AGE'] = 1999 - dataFrame['YEAR']
	dataFrame.YOJ = dataFrame.YOJ.fillna(dataFrame.YOJ.mean())

	dataFrame.INCOME = dataFrame.INCOME.fillna(0)
	dataFrame['INCOME'] = dataFrame['INCOME'].map(lambda x: str(x)[1:])
	dataFrame.INCOME = dataFrame.INCOME.fillna(0)
	dataFrame.INCOME = dataFrame.INCOME.replace(to_replace = "", value = 0)
	dataFrame.INCOME = dataFrame.INCOME.astype(int)

	dataFrame.HOME_VAL = dataFrame.HOME_VAL.fillna(0)
	dataFrame['HOME_VAL'] = dataFrame['HOME_VAL'].map(lambda x: str(x)[1:])
	dataFrame.HOME_VAL = dataFrame.HOME_VAL.replace(to_replace = '', value = 0)
	dataFrame.HOME_VAL = dataFrame.HOME_VAL.fillna(0)
	dataFrame.HOME_VAL = dataFrame.HOME_VAL.astype(int)

	#dataFrame.OCCUPATION = dataFrame.OCCUPATION.str.replace("", "None")

	dataFrame.BLUEBOOK = dataFrame.BLUEBOOK.fillna(0)
	dataFrame['BLUEBOOK'] = dataFrame['BLUEBOOK'].map(lambda x: str(x)[1:])
	dataFrame.BLUEBOOK = dataFrame.BLUEBOOK.replace(to_replace = '', value = 0)
	dataFrame.BLUEBOOK = dataFrame.BLUEBOOK.fillna(0)
	dataFrame.BLUEBOOK = dataFrame.BLUEBOOK.astype(int)

	dataFrame.OLDCLAIM = dataFrame.OLDCLAIM.fillna(0)
	dataFrame['OLDCLAIM'] = dataFrame['OLDCLAIM'].map(lambda x: str(x)[1:])
	dataFrame.OLDCLAIM = dataFrame.OLDCLAIM.replace(to_replace = '', value = 0)
	dataFrame.OLDCLAIM = dataFrame.OLDCLAIM.fillna(0)
	dataFrame.OLDCLAIM = dataFrame.OLDCLAIM.astype(int)

	dataFrame.CLM_AMT = dataFrame.CLM_AMT.fillna(0)
	dataFrame['CLM_AMT'] = dataFrame['CLM_AMT'].map(lambda x: str(x)[1:])
	dataFrame.CLM_AMT = dataFrame.CLM_AMT.replace(to_replace = '', value = 0)
	dataFrame.CLM_AMT = dataFrame.CLM_AMT.fillna(0)
	dataFrame.CLM_AMT = dataFrame.CLM_AMT.astype(int)
	# In the given data Most claim amounts hold at below 10000 which can be assumed minimum amount of loss if the conditions are met and accident occurs
	mean = dataFrame.loc[dataFrame['CLM_AMT'] < 9000, 'CLM_AMT'].mean()
	dataFrame.loc[dataFrame.CLM_AMT > 9000, 'CLM_AMT'] = mean
	dataFrame.CLM_AMT = dataFrame.CLM_AMT.astype(np.float32)
	dataFrame.CLM_AMT = dataFrame.CLM_AMT.fillna(0.0)
	#SdataFrame['ISLOST'] = dataFrame.apply(f, axis = 1)


	dataFrame.CAR_AGE = dataFrame.CAR_AGE.fillna(0)

	dataFrame.CAR_AGE = dataFrame.CAR_AGE.replace(to_replace = '', value = 0)
	dataFrame.CAR_AGE = dataFrame.CAR_AGE.fillna(0)
	dataFrame.CAR_AGE = dataFrame.CAR_AGE.astype(int)

	print("Primary dataframe cleanup is complete")


#  ##################Correlation Matrices#######################
def plot_correlation_data():
	global dataFrame
	corr_mat = dataFrame.corr()
	f, ax = plt.subplots(figsize = (12, 9))
	sns.heatmap(corr_mat, vmax = .8, square = True);
	plt.show()
	k = 11  # number of variables for heatmap
	cols = corr_mat.nlargest(k, 'CLM_AMT')['CLM_AMT'].index
	most_corr = pd.DataFrame(cols)
	most_corr.columns = ['Most Correlated Features']
	print(most_corr)


#  ################Analysis Graphs used to check the plots against claim amounts##############
def plt_analysis_graphs():
	plt.scatter(x = dataFrame.CLAIM_FLAG, y = dataFrame.CLM_AMT)
	plt.ylabel('Claim Amount')
	plt.xlabel('Claim Flag')
	plt.show()

	plt.scatter(x = dataFrame.AGE, y = dataFrame.CLM_AMT)
	plt.ylabel('Claim Amount')
	plt.xlabel('AGE')
	plt.show()

	plt.scatter(x = dataFrame.MVR_PTS, y = dataFrame.CLM_AMT)
	plt.ylabel('Claim Amount')
	plt.xlabel('MVR_PTS')
	plt.show()

	plt.scatter(x = dataFrame.CLM_FREQ, y = dataFrame.CLM_AMT)
	plt.ylabel('Claim Amount')
	plt.xlabel('Claim Frequency')
	plt.show()

	plt.scatter(x = dataFrame.HOMEKIDS, y = dataFrame.CLM_AMT)
	plt.ylabel('Claim Amount')
	plt.xlabel('HOME Kids')
	plt.show()

	plt.scatter(x = dataFrame.KIDSDRIV, y = dataFrame.CLM_AMT)
	plt.ylabel('Claim Amount')
	plt.xlabel('KIDSDRIV')
	plt.show()

	plt.scatter(x = dataFrame.YEAR, y = dataFrame.CLM_AMT)
	plt.ylabel('Claim Amount')
	plt.xlabel('YEAR')
	plt.show()

	plt.scatter(x = dataFrame.TRAVTIME, y = dataFrame.CLM_AMT, c = "Red")
	plt.ylabel('Claim Amount')
	plt.xlabel('TRAV TIME')
	plt.show()

	plt.scatter(x = dataFrame.BLUEBOOK, y = dataFrame.CLM_AMT, c = "Green")
	plt.ylabel('Claim Amount')
	plt.xlabel('BLUE BOOK')
	plt.show()

	plt.scatter(x = dataFrame.GENDER, y = dataFrame.CLM_AMT, c = "#00B8D4")
	plt.ylabel('Claim Amount')
	plt.xlabel('Gender')
	plt.show()


#  ###########Probe data to understand features and feature tyoes and understand total features of interest###########
def probe():
	global dataFrame
	print("DATA DESCRIPTION")
	print(dataFrame.describe())

	print("Data type Analysis for objects non numbers")
	print(dataFrame.select_dtypes(include = ['object']).columns)
	print(len(dataFrame.select_dtypes(include = ['object']).columns))

	cleanup_data(dataFrame)

	print(dataFrame.select_dtypes(include = ['object']).columns)
	cat = len(dataFrame.select_dtypes(include = ['object']).columns)
	num = len(dataFrame.select_dtypes(include = ['int64', 'float64']).columns)
	print('Total Features: ', cat, 'categorical', '+', num, 'numerical', '=', cat + num, 'features')

	print("########Correlation########")
	plot_correlation_data()
	print("######Analysis Graphs######")
	plt_analysis_graphs()



def normalise_scale_outliers():
	global dataFrame;
	# dataFrame.drop(dataFrame.columns.difference(['CLM_AMT', 'CLAIM_FLAG', 'MVR_PTS', 'CLM_FREQ', 'PARENT1', 'MSTATUS',
	#                                             'OLDCLAIM','REVOKED']), 1, inplace = True)
	dataFrame = pd.get_dummies(dataFrame, drop_first=True)

	#scaler = StandardScaler()
	#dataFrame[['CLAIM_FLAG', 'PARENT1_Yes', 'MSTATUS_z_No', 'REVOKED_Yes']] = scaler.fit_transform(dataFrame[['CLAIM_FLAG', 'PARENT1_Yes', 'MSTATUS_z_No', 'REVOKED_Yes']])
	# # ## TOBE SCALLED

	pass

'''
#SAIF CODE BEGINS HERE
def loss_or_not(x):
    if(x>1.0):
        return 1
    else:
        return 0
    

dataFrame['Loss_or_Not']=dataFrame['OLDCLAIM'].apply(loss_or_not)

df2=dataFrame.drop(columns=['OLDCLAIM'])

x=df2.values[:,:49]
y=df2.values[:,49]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

from sklearn.ensemble import BaggingClassifier
from sklearn import tree

model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(x_train, y_train)
preds=model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, preds)



df3=dataFrame.loc[dataFrame['Loss_or_Not']==1]
df3=df3.drop(columns=['Loss_or_Not'])
y2=df3['OLDCLAIM'].values()
df3=df3.drop(columns=['OLDCLAIM'])
x2=df3.values[:,:49]
x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size = 0.25, random_state = 42)

regressor = RandomForestRegressor(n_estimators =250, random_state = 0, n_jobs =5)
regressor.fit(x_train, y_train)
y_predicted = regressor.predict(x_test)

print(str(y_predicted))
mse = mean_squared_error(y_test, y_predicted)
rms = sqrt(mse)
print("MSE=" + str(mse) + "RMS VALUE" + str(rms))
print(r2_score(y_test, y_predicted))




#SAIF CODE ENDS HERE 
'''



def model_apply():
	global dataFrame
#	dataFrame = dataFrame[dataFrame.ISLOST != 0]

	Y = dataFrame.CLM_AMT
	dataFrame.drop(['CLM_AMT'], axis = 1, inplace = True)
	print(Y.describe())
	dataFrame.drop(['ID'], axis = 1, inplace = True)
	X = dataFrame
	print(np.where(np.isnan(Y)))


	X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 0.25, test_size = 0.70,
	                                                    random_state = 42)
	#del dataFrame
	del X, Y
	gc.collect()

	regressor = RandomForestRegressor(n_estimators =250, random_state = 0, n_jobs =5)
	regressor.fit(X_train, y_train)
	y_predicted = regressor.predict(X_test)

	print(str(y_predicted))
	mse = mean_squared_error(y_test, y_predicted)
	rms = sqrt(mse)
	print("MSE=" + str(mse) + "RMS VALUE" + str(rms))
	print(r2_score(y_test, y_predicted))

	pass


def main():
	probe()  # Function for data analysis
	normalise_scale_outliers()  # Function to encode and scale data.print("########Correlation########")
	#plot_correlation_data()
	model_apply()  # Function to apply the ML models and check for RMSE and R2 scores errors




if __name__ == '__main__':
	main()