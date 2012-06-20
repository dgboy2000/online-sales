DEBUG = True
NUM_FOLDS = 3
USE_DATA_CACHE = True

ENSEMBLE = [
    'LinearRegression',
    'QuantLinearRegression',
    'RidgeRegression',
    'RandomForest',
    'SupportVectorMachines',
    'GradientBoosting',
    'DecisionTree'
]

# ADD = [ 'DecisionTree' ]

FEATURE_CACHE = {
  'LinearRegression': True,
  'QuantLinearRegression': True,
  'SVMRegression': False,
  'RidgeRegression': True,
  'RandomForest': True
}









