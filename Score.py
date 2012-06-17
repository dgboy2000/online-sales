import math
import numpy as np

class Score:
  def __init__(self, sales, predictions):
    self.sales = sales
    self.predictions = predictions
    
  def getRMSLE(self):
    N = self.sales.shape[0] * self.sales.shape[1]
    SE = (self.sales - self.predictions) ** 2
    return math.sqrt( SE.sum() / N )
    
    
    
    
    
    
    
    
    
    
    
    """
    
import numpy as np
import math
from Score import Score
a=np.ones((2,2))
s=Score(a,a)
s.getRMSLE()
    """
    
    
