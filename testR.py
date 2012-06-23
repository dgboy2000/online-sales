
import rpy2.interactive as r
import rpy2.interactive as r
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
import scikits.learn.linear_model as lm
import scikits.learn
import numpy as np
import pickle
import marshal
from Run import Run

enet_path = lm.enet_path
from scikits.learn import datasets

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

X /= X.std(0) 
eps = 5e-3
print "Computing regularization path using the elastic net..."
models = enet_path(X, y, eps=eps, rho=0.8)
alphas_enet = np.array([model.alpha for model in models])
coefs_enet = np.array([model.coef_ for model in models])
glmnet = importr("glmnet")
predict = robjects.r["predict"]
plot = robjects.r["plot"]

run = Run()
run._setup()

dataset = run.ds_train
month_ind = 1
monthly_sales = dataset.getSalesForMonth(month_ind)
x = dataset.getFeaturesForMonth(month_ind)
x /= x.std(0) 
eps = 5e-3
print "Computing regularization path using the elastic net..."
models = enet_path(x, monthly_sales, eps=eps, rho=0.8)

mx = robjects.r.matrix(x,ncol=dataset.getNumFeatures())
y = robjects.vectors.FloatVector(monthly_sales)
fit1 = glmnet.glmnet(mx,y)


preds = glmnet.predict(fit1,mx)
m = robjects.r.matrix(robjects.IntVector(range(10)), nrow=5)
plot = robjects.r.plot
rnorm = robjects.r.rnorm
plot(rnorm(100), ylab="random") 
sample = robjects.r.sample
summary = robjects.r.summary
lm = robjects.r.lm
df = robjects.r['data.frame'](robjects.r['cbind'](y,x))
        
res = lm("y ~.",df)
x=robjects.r.matrix(rnorm(100*20),100,20)
y=rnorm(100)
g2=sample(range(1,3),100,replace=True)
g4=sample(range(1,5),100,replace=True)
fit1=glmnet.glmnet(x,y)
pickle.dump(fit1, open('tmp.pickle', 'w'))
obj2 = pickle.load(open('tmp.pickle', 'rb'))
print(obj2)

xdf = robjects.r.[data.frame](robjects.r[])

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