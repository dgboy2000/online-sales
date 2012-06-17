import math
import numpy as np

class Score:
  def __init__(self, sales, predictions):
    self.sales = sales
    self.predictions = predictions
    
  def getRMSLE(self):
    N = self.sales.shape[0] * self.sales.shape[1]
    
    log_sales = np.log(self.sales+1)
    log_preds = np.log(self.predictions+1)
    SLE = (log_sales - log_preds) ** 2
    
    return math.sqrt( SLE.sum() / N )
    
    
    
    
    
    
    
    
    
    
    
    """
    
import numpy as np
import math
from Score import Score
a=np.ones((2,2))
s=Score(a,a)
s.getRMSLE()
    """
    
    
