
# Check the versions of libraries
 
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

import statsmodels.api as sm

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset

names  =['InkTj','avgTrosbegrBer' ,'AvgPensAllm','Förseningsavgift' ,'Skuldsanering' ,'AntalFlyttar' ,'AdressTyp' ,'SkuldSaldoEMaxB' ,'SökandeBF' ,'SökandeBFAntal' ,'AntalPåAdress' ,'CO' ,'Kvinna' ,'Född','SföreB' ,'Y'] 
#names  =['InkTj','Y'] 
#names  =['age','totcap' ,'pay'] 
dataset = pandas.read_csv('SatagoSEk2_mod.csv', names=names)


#for column in dataset.columns:
#    if dataset[column].dtype == type(object):
#        le = preprocessing.LabelEncoder()
#        dataset[column] = le.fit_transform(dataset[column])

dataset_onehot = dataset.copy()
#dataset_onehot = pandas.get_dummies(dataset_onehot, columns=['Skuldsanering'], prefix = ['Skuldsanering'])
dataset_onehot = pandas.get_dummies(dataset_onehot, columns=['SökandeBF'], prefix = ['SökandeBF'])
dataset_onehot = pandas.get_dummies(dataset_onehot, columns=['Skuldsanering'], prefix = ['Skuldsanering'])


print(dataset_onehot.head())

dataset  =dataset_onehot.copy()

dataset = dataset[[c for c in dataset if c not in ['Y']] 
       + ['Y']]

for col in dataset.columns:
    print(col)

# shape
print(dataset.shape)

# head
print(dataset.head(20))

# descriptions
#print(dataset.describe())

# class distribution
#print(dataset.groupby('pay').size())

# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()


# histograms
#dataset.hist()
#plt.show()


# scatter plot matrix
#scatter_matrix(dataset)
#plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:22]
Y = array[:,22]
#X = array[:,0:1]
#Y = array[:,1]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Make predictions on validation dataset
lr = LogisticRegression(solver='liblinear', multi_class='ovr')
#lr = LogisticRegression(solver='liblinear', multi_class='ovr',class_weight='balanced')
lr.fit(X_train, Y_train)
print(lr.coef_)
print(lr.intercept_)

#probs=lr.predict_proba(X_train)
#print(probs)

#THRESHOLD = 0.25
#preds = numpy.where(lr.predict_proba(X_validation)[:,1] > THRESHOLD, 1, 0)

#logit_model=sm.Logit(Y_train,X_train)
#result=logit_model.fit()
#print(result.summary2())


#Xnew = [[24,1,1,1]]
# make a prediction
#ynewp =lr.predict_proba(Xnew)
#print("X=%s, Predicted=%s" % (Xnew[0], ynewp[0]))
#ynew =lr.predict(Xnew)
#print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
#predictions = lr.predict(X_validation)
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))

