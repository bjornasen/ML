# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas 


# load the dataset
#dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
#X = dataset[:,0:8]
#y = dataset[:,8]



names  =['InkTj','avgTrosbegrBer' ,'AvgPensAllm','Förseningsavgift' ,'Skuldsanering' ,'AntalFlyttar' ,'AdressTyp' ,'SkuldSaldoEMaxB' ,'SökandeBF' ,'SökandeBFAntal' ,'AntalPåAdress' ,'CO' ,'Kvinna' ,'Född','SföreB' ,'Y'] 
dataset = pandas.read_csv('SatagoSEk2_mod.csv', names=names)

dataset_onehot = dataset.copy()
dataset_onehot = pandas.get_dummies(dataset_onehot, columns=['SökandeBF'], prefix = ['SökandeBF'])
dataset_onehot = pandas.get_dummies(dataset_onehot, columns=['Skuldsanering'], prefix = ['Skuldsanering'])


#print(dataset_onehot.head())

dataset  =dataset_onehot.copy()

dataset = dataset[[c for c in dataset if c not in ['Y']] 
       + ['Y']]

for col in dataset.columns:
    print(col)

# shape
#print(dataset.shape)

# head
#print(dataset.head(20))

# Split-out validation dataset
array = dataset.values
X = array[:,0:22]
y = array[:,22]


# define the keras model
model = Sequential()
model.add(Dense(10, input_dim=22, activation='relu'))
model.add(Dense(4, activation='relu'))
#model.add(Dense(22, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=50, batch_size=10)

# evaluate the keras model
loss, accuracy = model.evaluate(X, y)
print('Loss: %.2f' % (loss*100))
print('Accuracy: %.2f' % (accuracy*100))

#model.save_weights("weights.txt")
#for layer in model.layers:
#g=layer.get_config()
#    h=layer.get_weights()
#print (g)
#    print (h)