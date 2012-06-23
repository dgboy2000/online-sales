from LearnerBase import LearnerBase
import numpy as np
from scipy import linalg
import Score
import rpy2.interactive as r
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from DataSet import DataSet
import scikits.learn.linear_model as lm
import scikits.learn
enet_path = lm.enet_path

class GlmNet(object):
    def __init__(self,debug=False):
        self.debug = debug
        self.params = None
        self.glmnet = importr("glmnet")
        self.glm_list = None
        
    def _train_with_values(self, dataset):
        pass
    
    def cross_validate(self, dataset, num_folds):
        pass
    
    def train(self, dataset):
        self.glm_list = []
        for month_ind in range(12):
            monthly_sales = dataset.getSalesForMonth(month_ind)
            x = dataset.getFeaturesForMonth(month_ind)
            #mx = robjects.r.matrix(x,ncol=dataset.getNumFeatures())
            #y = robjects.vectors.FloatVector(monthly_sales)
            x /= x.std(0) 
            eps = 5e-3
            print "Computing regularization path using the elastic net..."
            models = enet_path(x, monthly_sales, eps=eps, rho=0.8, alpha=0.5)
            self.glm_list.append(models[0])
    
    def predict(self, dataset):
        features = dataset.getFeatures()
        features /= features.std(0)
        num_samples, num_features = features.shape
        
        predictions = np.zeros((num_samples, 12))
        for month_ind in range(12):
            predictions[:, month_ind] = self.glm_list[month_ind].predict(features)
        return predictions
    
LearnerBase.register(GlmNet)
    
    
    
    
    
    
    
    
    
    
    
