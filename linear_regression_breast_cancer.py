import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression

os.chdir("C:/Users/sunan/Desktop/review example/linear")
data_all=pd.read_csv("./breast-cancer-wisconsin-data_data.csv",index_col=0)
data_features=data_all.iloc[:,1:11]
data_diagnosis_l=data_all.iloc[:,0]
#label convert to numeric
data_diagnosis=data_diagnosis_l.replace("B",0)
data_diagnosis=data_diagnosis.replace("M",1)
#####################
X=data_features.values
Y=data_diagnosis.values
clf = LogisticRegression(random_state=0, solver='lbfgs',class_weight="balanced").fit(X, Y)
P=clf.predict(X)
P_P=clf.predict_proba(X)
sc=clf.score(X, Y)
w_est=clf.coef_.reshape(-1,1)
b_est=clf.intercept_
eta=np.matmul(X,w_est)+b_est
result_df=pd.DataFrame(dict(label=Y, eta=eta.reshape(-1), prediction=P_P[:,1].reshape(-1)))
groups=result_df.groupby('label')
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.eta, group.prediction, marker='o', linestyle='',ms=3, label=name)
ax.legend(fontsize=15.0)
ax.set_xlabel(r'$\eta=Xw$',fontsize = 15.0)
ax.set_ylabel('probability of Y=M ',fontsize = 15.0)
plt.show()
##########using neural network