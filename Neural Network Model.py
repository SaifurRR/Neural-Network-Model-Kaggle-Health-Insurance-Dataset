#import libraries
import pandas as pd #df
import tensorflow #ML
from sklearn.model_selection import train_test_split #split to test_size
from sklearn.compose import ColumnTransformer #apply Scaling
from sklearn.preprocessing import StandardScaler #mean: 0, var:1
from tensorflow.keras.models import Sequential #NN seq: i/p, hidden layer, o/p 
from tensorflow.keras.layers import InputLayer #takes nodes/features
from tensorflow.keras.layers import Dense #Hidden layer
from tensorflow.keras.optimizers import Adam #Optimizer: SGD, Softmax

print(tensorflow.random.set_seed(35)) #for the reproducibility of results

#function that returns NN Sequential model:
def design_model(features):
  #instantiate NN model
  model = Sequential(name = "my_first_model")
  #InputLayer() : placeholder for i/p data
  input = InputLayer(input_shape=(features.shape[1],)) 
  #add the i/p layer to model
  model.add(input) 
  #add hidden layer with 128 neurons to handle complex dependencies
  model.add(Dense(128, activation='relu')) 
  #o/p layer: predicts 1 value in Regression
  model.add(Dense(1)) 
  #optimizer (Adam): learning rate for each features/nodes
  opt = Adam(learning_rate=0.1)
  #measure of learning, loss: MSE, metrics: MAE
  model.compile(loss='mse',  metrics=['mae'], optimizer=opt)
  return model
#load the dataset
dataset = pd.read_csv('insurance.csv') 
#features/nodes
features = dataset.iloc[:,0:6]
#labels/output
labels = dataset.iloc[:,-1] 
#one-hot encoding
features = pd.get_dummies(features)  
#train-test split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42) 
 
#standardize
ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')
#fit-transform: training data -> calc. new mean & var
features_train = ct.fit_transform(features_train)
#transform: test data -> apply calculated mean and var  from 'train-data'
features_test = ct.transform(features_test)

#call NN Seq, function for our model design
model = design_model(features_train)
print(model.summary())

#fit the model using 40 epochs and batch size 1
model.fit(features_train,labels_train, epochs=40, batch_size=1, verbose=1)

#evaluate the model on the test data
val_mse, val_mae = model.evaluate(features_test,labels_test,verbose=0)
 
print("MAE: ", val_mae)
print("MSE: ", val_mse)


