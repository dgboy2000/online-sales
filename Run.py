import csv
import DataSet
import learn
import params
import random
import Score

class Run:
  def __init__(self):
    self.ds_test = None
    self.ds_train = None
    self.ensemble = None
    self.train_probs = None
    self.test_probs = None
    
  def _eval(self):
    train_score = Score.Score(self.ds_train.getLabels(), self.train_probs)
    # test_score = Score.Score(self.ds_test.getLabels(), self.test_probs)    
    
    print "Train Score %f" %train_score.getLogLoss()
    # print "Test Score %f" %test_score.getLogLoss()
    
  def _predict(self):
    if params.DEBUG:
      print "Running prediction..."
    self.train_probs = self.ensemble.predict(self.ds_train)
    self.test_probs = self.ensemble.predict(self.ds_test)
      
  def _setup(self):
    if params.DEBUG:
      print "Reading training data..."
    self.ds_train = DataSet.DataSet(True)
    self.ds_train.importData('data/train.csv')

    if params.DEBUG:
      print "Reading test data..."
    self.ds_test = DataSet.DataSet(False)
    self.ds_test.importData('data/test.csv')
    
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
    data_out = csv.writer(open('output/kaggle.csv','w'))
    for prob in self.test_probs:
      data_out.writerow([prob])
    
    
if __name__ == '__main__':
  random.seed(0)
  run = Run()
  run.run()
  run.outputKaggle()
    
    
    
    
    
    
    
    
    