import csv
import DataSet
import learn
import math
import os
import params
import random
import Score

class Run:
  def __init__(self):
    self.ds_test = None
    self.ds_train = None
    self.ensemble = None
    self.train_sales = None
    self.test_sales = None
    
  def _eval(self):
    train_score = Score.Score(self.ds_train.getSales(), self.train_sales)
    # test_score = Score.Score(self.ds_test.getSales(), self.test_sales)    
    
    print "Train Score %f" %train_score.getRMSLE()
    # print "Test Score %f" %test_score.getRMSLE()
    
  def _predict(self):
    if params.DEBUG:
      print "Running prediction..."
    self.train_sales = self.ensemble.predict(self.ds_train)
    self.test_sales = self.ensemble.predict(self.ds_test)
      
  def _setup(self):
    if params.DEBUG:
      print "Reading training data..."
    self.ds_train = DataSet.DataSet(True)
    self.ds_train.importData('data/train.csv')

    if params.DEBUG:
      print "Reading test data..."
    self.ds_test = DataSet.DataSet(False)
    self.ds_test.importData('data/test.csv')
    self.ds_test.dropUselessFeatures(self.ds_train.getUselessFeatures())
    
  def _train(self):
    if params.DEBUG:
      print "Training ensemble..."
    self.ensemble = learn.Ensemble.Ensemble(params.DEBUG)
    self.ensemble.addLearner(learn.LinearRegression.LinearRegression(debug=params.DEBUG))
    # self.ensemble.addLearner(learn.RidgeRegression.RidgeRegression(debug=params.DEBUG))
    # self.ensemble.addLearner(learn.RandomForest.RandomForest(debug=params.DEBUG))
    self.ensemble.train(self.ds_train, params.NUM_FOLDS)
          
  def run(self):
    self._setup()
    self._train()
    self._predict()
    self._eval()
      
  def outputKaggle(self):
    if not os.path.exists('output'):
      os.mkdir('output')

    data_out = csv.writer(open('output/kaggle.csv','w'))
    
    headers = ['id']
    for i in range(12):
      headers.append("Outcome_M%d" %(i+1))
    data_out.writerow(headers)
    
    for ind,sales in enumerate(self.test_sales):
      exp_sales = [math.exp(val) for val in sales]
      data_out.writerow([self.ds_test.ids[ind]] + exp_sales)
    
    
if __name__ == '__main__':
  random.seed(0)
  run = Run()
  run.run()
  run.outputKaggle()
    
    
    
    
    
    
    
    
    
