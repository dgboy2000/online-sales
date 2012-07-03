from scipy import linalg

class SVDFeatureCompressor:
  def __init__(self):
    self.features = None
    self.num_eigenvectors = None
    self.P = None
    
  def computeSVD(self, features, num_eigenvectors):
    """Compute SVD approximation on the specified features, retaining only the projection onto the
    the top num_eigenvectors singular values."""
    U, s, Vh = linalg.svd(features)
    
    num_samples, num_features = features.shape
    s[num_eigenvectors:] = 0.0
    S = linalg.diagsvd(s, num_samples, num_features)
    
    L = S.dot(Vh)
    L = L[:num_eigenvectors, :]
    
    self.P = L.transpose().dot(linalg.inv(L.dot(L.transpose())).dot(L)).transpose()
    
  def projectSVD(self, features):
    return features.dot(self.P)
    
  






