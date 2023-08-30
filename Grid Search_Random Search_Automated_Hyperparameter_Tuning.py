#import gridsearch
from sklearn.model_selection import GridSearchCV
#import randomized search
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint as sp_randint
# wrap NN model to KerasRegressor
from keras.wrappers.scikit_learn import KerasRegressor
#make scorer in GridSearchCV
from sklearn.metrics import mean_squared_error
#scoring in GridSearchCV
from sklearn.metrics import make_scorer
#import model function 
from model import design_model, features_train, labels_train

#------------- GRID SEARCH --------------
def do_grid_search():
  # list of batch to try
  batch_size = [6, 64]
  # list of epochs
  epochs = [10, 50]
  # wrap NN model to 'KerasRegressor', takes NN model function or any model as arg, e.g., RandomForest.
  model = KerasRegressor(build_fn=design_model)
  # dict to use in param_grid
  param_grid = dict(batch_size=batch_size, epochs=epochs)
  # 
  grid = GridSearchCV(estimator = model, param_grid=param_grid, scoring = make_scorer(mean_squared_error, greater_is_better=False), return_train_score = True)
  # train on gridsearch
  grid_result = grid.fit(features_train, labels_train, verbose = 0)
  print(grid_result)
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  print(dir(grid_result.cv_results_))
  print("Test")
  #extract test scores
  means = grid_result.cv_results_['mean_test_score']
  #extract std
  stds = grid_result.cv_results_['std_test_score']
  #extract param: batch size and epochs
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))

  print("Traininig")
  #extract train scores
  means = grid_result.cv_results_['mean_train_score']
  stds = grid_result.cv_results_['std_train_score']
  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))

#------------- RANDOMIZED SEARCH --------------
def do_randomized_search():
  #sp_randint(low, high) -> 
  param_grid = {'batch_size': sp_randint(2, 16), 'nb_epoch': sp_randint(10, 100)}
  model = KerasRegressor(build_fn=design_model)
  grid = RandomizedSearchCV(estimator = model, param_distributions=param_grid, scoring = make_scorer(mean_squared_error, greater_is_better=False), n_iter = 12)
  grid_result = grid.fit(features_train, labels_train, verbose = 0)
  print(grid_result)
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))

print("-------------- GRID SEARCH --------------------")
do_grid_search()
print("-------------- RANDOMIZED SEARCH --------------------")
do_randomized_search()

