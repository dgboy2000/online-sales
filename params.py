DEBUG = True
NUM_FOLDS = 4
USE_DATA_CACHE = True

# Data transformations
LOG_TRANSFORM = False
STANDARDIZE_DATA = True

ENSEMBLE = [
  # 'LinearRegression',
  # 'QuantLinearRegression',
  'RidgeRegression',
  'RandomForest',
  'SupportVectorMachines',
  'GradientBoosting',
  # 'DecisionTree',
  # 'NearestNeighbor',
  'GlmNet'
]

# ENSEMBLE = [ 'DecisionTree' ]

FEATURE_CACHE = {
  'LinearRegression': True,
  'QuantLinearRegression': True,
  'SVMRegression': False,
  'RidgeRegression': True,
  'GradientBoosting': True,
  'RandomForest': True,
  'NearestNeighbor': True,
  'SupportVectorMachines': True,
  'GlmNet': True
}









