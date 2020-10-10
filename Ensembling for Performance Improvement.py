import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,GridSearchCV
import time
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from scipy import interpolate

df=pd.read_csv('ccdefault.csv',index_col='ID')
df.head()

df.dropna()

X=df.iloc[:,0:23].values
y=df.iloc[:,23].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=33, stratify=y)
#part 1
rf=RandomForestClassifier(n_estimators=50,criterion='gini',random_state=1,n_jobs=-1)
pipe=Pipeline([['sc',StandardScaler()],['randomforest',rf]])
params={'randomforest__n_estimators':[20,50,75,90,100]}
grid=GridSearchCV(estimator=pipe,param_grid=params,cv=2)#,scoring='roc_auc'
grid.fit(X_train,y_train)
scores=cross_val_score(grid,X_train,y_train,scoring='accuracy',cv=5)
y_pred=grid.predict(X_test)
results=grid.cv_results_
print('')
print('GridSearch:')
print('Tuned Model Parameters:{}'.format(grid.best_params_))
#print('In-sample Accuracy:%.4f'% grid.best_score_)
print('In-sample CV Accuracy:%.4f +/- %.4f'% (np.mean(scores),np.std(scores)))
forest=grid.best_estimator_
forest.fit(X_train,y_train)
print('Out-Sample Accuracy:%.4f'% grid.score(X_test,y_test))

#Explore the relationship between n_estimators, in-sample CV accuracy and computation time
print('Relationship between n_estimators and in-sample accuracy:')
for i in range(5):
    print('n_estimators:',params['randomforest__n_estimators'][i],
          "  in-sample accuracy",'%.4f'% results['mean_test_score'][i])
print('Relationship between n_estimators and computation time:')
for i in range(5):
    print('n_estimators:',params['randomforest__n_estimators'][i],
          "  computation time",'%.4f'%(results['mean_fit_time'][i]+results['mean_score_time'][i]))
#plot the Relationship between n_estimators and in-sample accuracy
xnew =np.arange(params['randomforest__n_estimators'][0],params['randomforest__n_estimators'][4],1)
func = interpolate.interp1d(params['randomforest__n_estimators'],results['mean_test_score'],kind='cubic')
ynew = func(xnew)
plt.figure(figsize=(7,5))
plt.plot(xnew,ynew)
plt.xlabel('n_estimators')
plt.ylabel('in-sample accuracy')
plt.title('Relationship between n_estimators and in-sample accuracy')
plt.show()

#part 2
fo=forest.__getitem__('randomforest')
feat_labels = df.columns[0:23]
importances = fo.feature_importances_
indices = np.argsort(importances)[::-1]
print('')
print('Feature importances:')
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.figure(figsize=(10,5))
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center')
plt.title('Features Importances')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
#plt.savefig('images/04_09.png', dpi=300)
plt.show()

'''
importances = pd.Series(data=forest.feature_importances_,index=feat_labels)
importances_sorted = importances.sort_values(ascending=False)
plt.figure(figsize=(12,5))
importances_sorted.plot(kind="bar")
plt.title('Features Importances')
plt.show()
print(importances)
'''

print("My name is {Xin Zhang}")
print("My NetID is: {xzhan81}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")