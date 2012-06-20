DEBUG = True
NUM_FOLDS = 3
USE_DATA_CACHE = True

ADD = [
    'LinearRegression',
    'RidgeRegression',
    'RandomForest',
    'SupportVectorMachines',
    'GradientBoosting',
    'DecisionTree'
]

# ADD = [ 'DecisionTree' ]

FEATURE_CACHE = {
  'LinearRegression': True,
  'SVMRegression': False,
  'RidgeRegression': True,
  'RandomForest': True
}









