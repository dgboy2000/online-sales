#!/usr/local/bin/python
import csv
import DataSet
import learn
import math
import os
import params
import pickle
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
    # TODO: separate a train and test dataset for ourselves and report on those numbers
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
    self.ds_train = self._readOrLoadDataset('train')
    self.ds_test = self._readOrLoadDataset('test', useless_features = self.ds_train.getUselessFeatures())

  def _readOrLoadDataset(self, ds_type, useless_features = None):
    fname = "cache/%s_data.pickle" %ds_type
    try:
      if not params.USE_DATA_CACHE:
        raise("Do not use cache")
      f = open(fname, 'rb')
      ds = pickle.load(f)
      if params.DEBUG:
        print "Using cached %s..." %fname
    except:
      if params.DEBUG:
        print "Reading and dumping %s..." %fname
        
      data_fname = "data/%s.csv" %ds_type
      ds = DataSet.DataSet(ds_type == 'train')
      ds.importData(data_fname)
      if useless_features is not None:
        ds.dropUselessFeatures(useless_features)
      pickle.dump(ds, open(fname, 'w'))

    return ds
    
  def _train(self):
    if params.DEBUG:
      print "Training ensemble..."
    self.ensemble = learn.Ensemble.Ensemble(params.DEBUG)

    self.ensemble.addLearner(learn.GlmNet.GlmNet(debug=params.DEBUG))


    for learner_class in params.ENSEMBLE:
      self.ensemble.addLearner(eval("learn.%s.%s" %(learner_class, learner_class))(debug=params.DEBUG))


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
    
    
    
    
    
    
    
    
    
