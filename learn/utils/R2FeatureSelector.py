import numpy as np
from scipy.stats import linregress

class R2FeatureSelector:
  def __init__(self):
    self.features = None
    self.target = None
    self.selected_feat_inds = None
  
  def selectVariables(self, features, target, num_variables=100, p_cutoff=0.05):
#    if feature is not binary, transform into matrix of indicators for each category
    
    """Find the top variables by correlation with target. Reject those below a p-value cutoff."""
    
    feat_regressions = []
    
    num_samples, num_features = features.shape
    for feat_ind in range(num_features):
      slope, intercept, r_value, p_value, std_err = linregress(features[:, feat_ind], y=target)
      feat_regressions.append([feat_ind, slope, r_value, p_value])

    feat_regressions.sort(key=lambda data_list: -data_list[2])
    
    top_variables = feat_regressions[:num_variables]
    self.selected_feat_inds = [data[0] for data in top_variables if data[3] <= p_value]

  def getSelectedVariables(self, features):
    return features[:, self.selected_feat_inds]
    
    
    
  


