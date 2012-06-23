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

class GlmNet(object):
    def __init__(self,debug=False):
        self.debug = debug
        self.params = None
        self.glmnet = importr("glmnet")
        
    def _train_with_values(self, dataset):
        self.glm_list = []
        for month_ind in range(12):
            monthly_sales = dataset.getSalesForMonth(month_ind)
            x = dataset.getFeaturesForMonth(month_ind)
            mx = robjects.r.matrix(x,ncol=dataset.getNumFeatures())
            y = robjects.vectors.FloatVector(monthly_sales)
            
            if self.debug:
                print "Learning on month %d of 12 with %d samples..." %(month_ind+1, len(monthly_sales))
                fit = self.glmnet.glmnet(mx,y)
                self.glm_list.append(fit)
    
    def cross_validate(self, dataset, num_folds):
        dataset.createFolds(num_folds)
        best_params = None
        best_rmsle = float("inf")
        cur_score = Score.Score()
        
        for fold_ind in range(num_folds):
            fold_train = dataset.getTrainFold(fold_ind)
            fold_test = dataset.getTestFold(fold_ind)
            self._train_with_values(fold_train)
            cur_score.addFold(fold_test.getSales(), self.predict(fold_test))
            
            cur_rmsle = cur_score.getRMSLE()
            if cur_rmsle < best_rmsle:
                if self.debug:
                    print "Achieved new best score %f" %cur_rmsle
                best_rmsle = cur_rmsle
    
    def train(self, dataset):
        if self.debug:
            print "Training glmnet "
            self._train_with_values(dataset)
    
    def predict(self, dataset):
        features = dataset.getFeatures()
        num_samples, num_features = features.shape
        
        predictions = np.zeros((num_samples, 12))
        for month_ind in range(12):
            predictions[:, month_ind] = self.glm_list[month_ind].predict(features)
        return predictions
    
LearnerBase.register(GlmNet)
    
    
    
    
    
    
    
    
    
    
    
