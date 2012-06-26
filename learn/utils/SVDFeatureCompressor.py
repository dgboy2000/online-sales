from scipy.linalg import svd

class SVDFeatureCompressor:
  def __init__(self):
    self.features = None
    self.num_eigenvectors = None
    
  def computeSVD(self, features, num_eigenvectors):
    """Compute SVD approximation on the specified features, retaining only the top num_eigenvectors
    singular values."""
    U, s, Vh = svd(features)
    
  def performSVD(self, features):
    pass






