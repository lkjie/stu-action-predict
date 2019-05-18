import statsmodels.api as sm
from sklearn.datasets import load_breast_cancer
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
# X1 = np.concatenate((X, np.ones((X.shape[0], 1))) ,axis=1)
X1 = np.random.randint(-10, 10, size=X.shape)
model = sm.Logit(y ,X1)
model = model.fit()
print(model.summary())