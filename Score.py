import math
import numpy as np

class Score:
  def __init__(self, sales=None, predictions=None):
    self.sales = sales
    self.predictions = predictions
    
  def addFold(self, sales, predictions):
    if self.sales is None:
      self.sales = sales
      self.predictions = predictions
    else:
      self.sales = np.vstack((self.sales, sales))
      self.predictions = np.vstack((self.predictions, predictions))
  
  def getRMSLE(self, month_ind=None):    
    N = 0
    SE = 0
    num_samples, num_sales = self.sales.shape

    if month_ind:
      for i in range(num_samples):
        if self.sales[i, month_ind] > 0:
          N += 1
          SE += (self.sales[i, month_ind] - self.predictions[i, month_ind]) ** 2
    else:
      for i in range(num_samples):
        for j in range(num_sales):
          if self.sales[i, j] > 0:
            N += 1
            SE += (self.sales[i, j] - self.predictions[i, j]) ** 2
    
    return math.sqrt( SE / N ) if N > 0 else 0.
    
    
    
    
    
    
    
    
    
    
    
    """
    
import numpy as np
import math
from Score import Score
a=np.ones((2,2))
s=Score(a,a)
s.getRMSLE()
    """
    
    
