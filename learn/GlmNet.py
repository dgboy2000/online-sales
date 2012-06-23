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

class GlmNet(object):
    alpha_values = [(i+1) * .01 for i in range(10)]
    
    def __init__(self,debug=False):
        self.debug = debug
        self.params = None
        self.glmnet = importr("glmnet")
        self.glm_list = None
        
        self.best_rmsle_list = [float("inf")] * 12
        self.best_alpha_list = [0] * 12
        
    def _train(self, dataset, alpha, alpha_list = None):
        self.glm_list = []
        for month_ind in range(12):
            monthly_sales = dataset.getSalesForMonth(month_ind)
            x = dataset.getFeaturesForMonth(month_ind)
            
            std = x.std(0)
            for i, v in enumerate(std):
                if v > 0:
                    x[:,i] /= v
                else:
                    x[:, i] = 0

            eps = 5e-3
            # print "Computing regularization path using the elastic net..."
            
            if alpha_list:
                alpha = alpha_list[month_ind]
            model = lm.ElasticNet(rho=0.8, alpha=alpha)
            
            print len(x), len(x[0]), len(monthly_sales)
            model.fit(x, monthly_sales)
            self.glm_list.append(model)
    
    def cross_validate(self, dataset, num_folds):
        dataset.createFolds(num_folds)
        for alpha in GlmNet.alpha_values:
            print "alpha = %f" % alpha
            score = Score.Score()
            for fold_ind in range(num_folds):
                fold_train = dataset.getTrainFold(fold_ind)
                fold_test = dataset.getTestFold(fold_ind)
                
                
                self._train(fold_train, alpha)
                
                print "sale: %s" % fold_test.getSales()
                print "predict: %s" % self.predict(fold_test)
                score.addFold(fold_test.getSales(), self.predict(fold_test))
            print score
                
            for month_ind in range(12):
                cur_rmsle = score.getRMSLE(month_ind)
                print "month_ind %d: cur_rmsle: %f" % (month_ind, cur_rmsle)
                
                if cur_rmsle < self.best_rmsle_list[month_ind]:
                    self.best_rmsle_list[month_ind] = cur_rmsle
                    self.best_alpha_list[month_ind] = alpha
                    
            print "best rmsle: %s" % str(self.best_rmsle_list)
            print "best alpha: %s" % str(self.best_alpha_list)
            
    def train(self, dataset):
        print "training on all data"
        self._train(dataset, None, self.best_alpha_list)
    
    def predict(self, dataset):
        features = dataset.getFeatures()
        
        std = features.std(0)
        for i, v in enumerate(std):
            if v > 0:
                features[:,i] /= v
            else:
                features[:, i] = 0
        num_samples, num_features = features.shape
        
        predictions = np.zeros((num_samples, 12))
        for month_ind in range(12):
            predictions[:, month_ind] = self.glm_list[month_ind].predict(features)
        return predictions
    
LearnerBase.register(GlmNet)
    
    
    
    
    
    
    
    
    
    
    
