import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split

def loss_or_not(x):
    if(x>1.0):
        return 1
    else:
        return 0


dataFrame = pd.read_csv('car_insurance_claim_2.csv', delimiter = ',')    

dataFrame['Loss_or_Not']=dataFrame['OLDCLAIM'].apply(loss_or_not)

df2=dataFrame.drop(columns=['OLDCLAIM'])

x=df2.values[:,:49]
y=df2.values[:,49]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)


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
