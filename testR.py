
import rpy2.interactive as r
import rpy2.interactive as r
import rpy2.robjects as robjects
import pickle
import marshal

glmnet = r.importr("glmnet")

m = robjects.r.matrix(robjects.IntVector(range(10)), nrow=5)
plot = robjects.r.plot
rnorm = robjects.r.rnorm
plot(rnorm(100), ylab="random") 
sample = robjects.r.sample

x=robjects.r.matrix(rnorm(100*20),100,20)
y=rnorm(100)
g2=sample(range(1,3),100,replace=True)
g4=sample(range(1,5),100,replace=True)
fit1=glmnet.glmnet(x,y)
pickle.dump(fit1, open('tmp.pickle', 'w'))
obj2 = pickle.load(open('tmp.pickle', 'rb'))
print(obj2)


def serialize_data(data, fname):
  """
  Writes `data` to a file named `fname`
  """
  with open(fname, 'wb') as f:
    marshal.dump(data, f)

def unserialize_data(fname):
  """
  Reads a pickled data structure from a file named `fname` and returns it
  IMPORTANT: Only call marshal.load( .. ) on a file that was written to using marshal.dump( .. )
  marshal has a whole bunch of brittle caveats you can take a look at in teh documentation
  It is faster than everything else by several orders of magnitude though
  """
  with open(fname, 'rb') as f:
    return marshal.load(f)

serialize_data(fit1,"tmp.marshall")