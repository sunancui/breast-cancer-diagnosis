import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
#set path to data folder
os.chdir("C:/Users/sunan/Desktop/review example/linear")
data_all=pd.read_csv("./breast-cancer-wisconsin-data_data.csv",index_col=0)
#read features
data_features=data_all.iloc[:,1:11]
#read labels
data_diagnosis_l=data_all.iloc[:,0]
#label convert to numeric
data_diagnosis=data_diagnosis_l.replace("B",0)
data_diagnosis=data_diagnosis.replace("M",1)
#get values of X, Y
X=data_features.values
Y=data_diagnosis.values
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
#################sklearn logistic regression
#by default this setting implement l2 penalty,
clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X, Y)
P=clf.predict(X)
P_P=clf.predict_proba(X)
sc=clf.score(X, Y)
#estimated optimal parameters
w_est=clf.coef_.reshape(-1,1)
b_est=clf.intercept_
eta=np.matmul(X,w_est)+b_est
result_df=pd.DataFrame(dict(label=Y, eta=eta.reshape(-1), prediction=P_P[:,1].reshape(-1)))
groups=result_df.groupby('label')
#plotting the result
fig, ax = plt.subplots()
ax.margins(0.05)
for name, group in groups:
    ax.plot(group.eta, group.prediction, marker='o', linestyle='',ms=3, label=name)
ax.legend(fontsize=15.0)
ax.set_xlabel(r'$\eta=Xw$',fontsize = 15.0)
ax.set_ylabel('probability of Y=M ',fontsize = 15.0)
plt.show()
#####################pytorch logistic regression
input_dim=X.shape[1]
output_dim=1
learning_rate=0.1
num_epoch=1000
#convert to pytorch tensor
X_torch=torch.from_numpy(X)
Y_torch=torch.from_numpy(Y.reshape(-1,1))
#define the model
class LogisticRg(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRg, self).__init__()
        self.linear=torch.nn.Linear(input_dim,output_dim)
    def forward(self,x):
        outputs=torch.sigmoid(self.linear(x))
        return outputs
model=LogisticRg(input_dim,output_dim)
criterion=torch.nn.BCELoss(reduction='sum')
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
for i in range(num_epoch):
    model.train()
    optimizer.zero_grad()
    y_pred=model(X_torch.float())
    #define the loss, adding l2 penalty
    loss=criterion(y_pred,Y_torch.float())
    #this is for matching the sklearn default setting, l2 penalty
    for l_w in model.parameters():
        loss=loss+torch.sum(l_w**2)*0.5
    loss.backward()
    optimizer.step()
    if i%10==0:
        print(i, loss)
#final estimated, note this is roughly the same result as sklearn implemention, the
#slight difference may comes from numerical problems, different optimizer, learning rate.
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)
