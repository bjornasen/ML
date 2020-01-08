from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

#try:
  # %tensorflow_version only exists in Colab.
#  %tensorflow_version 2.x
#except Exception:
#  pass
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


#dataframe = pd.read_csv('RAWDATA191119_head.csv')
dataframe = pd.read_csv('BIG_ONE.csv')
#print(dataframe.head())

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('pay')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds



feature_columns = []
categorical_columns = []

# numeric cols
for header in ['age']:
  feature_columns.append(feature_column.numeric_column(header))

age = feature_column.numeric_column("age")

# bucketized cols
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
#feature_columns.append(age_buckets)

#for header in ['KreditFF','Insolvent','Reglerad','man']:
  #categorical_column =feature_column.categorical_column_with_identity(key=header, num_buckets=100, default_value=0)
  #categorical_columns.append(feature_column.categorical_column_with_identity(key=header, num_buckets=100, default_value=0))

categorical_column_1 = feature_column.categorical_column_with_identity(key='KreditFF360', num_buckets=100, default_value=0)
categorical_column_2 = feature_column.categorical_column_with_identity(key='A_Insolvent360', num_buckets=100, default_value=0)
categorical_column_3 = feature_column.categorical_column_with_identity(key='A_Utslag360', num_buckets=100, default_value=0)
categorical_column_4 = feature_column.categorical_column_with_identity(key='man', num_buckets=2, default_value=0)


feature_columns =[
     tf.feature_column.indicator_column(categorical_column_1),
     tf.feature_column.indicator_column(categorical_column_2),
     tf.feature_column.indicator_column(categorical_column_3)
     #tf.feature_column.indicator_column(categorical_column_m)
]

#feature_columns.append(age_buckets)




print (feature_columns)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

          
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)


##### ESTIMATOR

feature_columns = []
categorical_columns.append(feature_column.categorical_column_with_identity(key='Utmatning360', num_buckets=100, default_value=0))
categorical_columns.append(feature_column.categorical_column_with_identity(key='A_Insolvent360', num_buckets=100, default_value=0))
categorical_columns.append(feature_column.categorical_column_with_identity(key='A_Utslag360', num_buckets=100, default_value=0))
categorical_columns.append(feature_column.categorical_column_with_identity(key='KreditFF360', num_buckets=100, default_value=0))
categorical_columns.append(feature_column.categorical_column_with_identity(key='KreditFF360_Betald', num_buckets=100, default_value=0))
categorical_columns.append(feature_column.categorical_column_with_identity(key='Firma360', num_buckets=100, default_value=0))
categorical_columns.append(feature_column.categorical_column_with_identity(key='Godman360', num_buckets=100, default_value=0))
categorical_columns.append(feature_column.categorical_column_with_identity(key='Egendom360', num_buckets=100, default_value=0))
categorical_columns.append(feature_column.categorical_column_with_identity(key='Bedrageri360', num_buckets=100, default_value=0))
categorical_columns.append(feature_column.categorical_column_with_identity(key='SS_Frivillig360', num_buckets=100, default_value=0))
categorical_columns.append(feature_column.categorical_column_with_identity(key='SS_Inledd360', num_buckets=100, default_value=0))
categorical_columns.append(feature_column.categorical_column_with_identity(key='SS_Avslutad360', num_buckets=100, default_value=0))
categorical_columns.append(feature_column.categorical_column_with_identity(key='man', num_buckets=10, default_value=0))


#for i in range (0,12):

feature_columns = []
feature_columns.append(tf.feature_column.indicator_column(categorical_columns[0]))
feature_columns.append(tf.feature_column.indicator_column(categorical_columns[1]))
#feature_columns.append(tf.feature_column.indicator_column(categorical_columns[2]))
feature_columns.append(tf.feature_column.indicator_column(categorical_columns[3]))
#feature_columns.append(tf.feature_column.indicator_column(categorical_columns[4]))

#feature_columns.append(tf.feature_column.indicator_column(categorical_columns[10]))
#feature_columns.append(age_buckets)
#feature_columns.append(age)


classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[32, 10],
    # The model must choose between 2 classes.
    n_classes=2)


classifier.train(
    input_fn=lambda:df_to_dataset(train, batch_size=32),
    steps=5000)


#classifier.train(
#    input_fn=lambda: input_fn(train, train_y, training=True),
#    steps=5000)


eval_result = classifier.evaluate(
    input_fn=lambda:df_to_dataset(test, shuffle=False,batch_size=32))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

print(eval_result)




