from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas 
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns




#names  =['KreditFF','Utslag','Insolvent','Utmätning','SS_Inledd','SS_Avslutad','Firma','SS_Frivillig','Bedrägeri','Godman','B2B_snartKonkurs','Egendom','Reglerad','pay'] 
#names  =['mindate','KreditFF','Utslag','Insolvent','Utmätning','SS_Inledd','SS_Avslutad','Firma','SS_Frivillig','Bedrägeri','Godman','Reglerad','pay'] 
#dataset = pandas.read_csv('satagofiUtfall2017.csv', names=names)

names  =['KreditFF','Insolvent','Reglerad','man','ålder','pay'] 
dataset = pandas.read_csv('RAWDATA191119.csv', names=names)


#dataset_onehot = dataset.copy()
#dataset_onehot = pandas.get_dummies(dataset_onehot, columns=['SökandeBF'], prefix = ['SökandeBF'])
#dataset_onehot = pandas.get_dummies(dataset_onehot, columns=['Skuldsanering'], prefix = ['Skuldsanering'])


#print(dataset_onehot.head())

#dataset  =dataset_onehot.copy()

#dataset = dataset[[c for c in dataset if c not in ['Y']] 
#       + ['Y']]

for col in dataset.columns:
    print(col)

# shape
#print(dataset.shape)

# head
#print(dataset.head(20))

# Split-out validation dataset
array = dataset.values
X = dataset.iloc[:,0:5]
y = dataset.iloc[:,5]

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X,y)
dfscores = pandas.DataFrame(fit.scores_)
dfcolumns = pandas.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pandas.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(11,'Score'))  #print 10 best features


model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pandas.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
#plt.show()

#names  =['mindate','KreditFF','Utslag','Insolvent','Utmätning','SS_Inledd','SS_Avslutad','Firma','SS_Frivillig','Bedrägeri','Godman','Reglerad','pay'] 
#data = pandas.read_csv('satagofiUtfall2017.csv',names=names)
names  =['KreditFF','Insolvent','Reglerad','man','ålder','pay'] 
data = pandas.read_csv('RAWDATA191119.csv', names=names)

X = data.iloc[:,0:5]  #independent columns
y = data.iloc[:,5]    #target column i.e price range
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(5,5))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

plt.show()