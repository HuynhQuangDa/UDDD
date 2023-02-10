import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

dt=pd.read_csv("DiamondsPrices2022.csv",index_col=0)

X=dt.iloc[:,[0,4,5,7,8,9]]
X
#cot nhan
Xnhan=dt.price
#su tuong quan giua độ sau va gia kim cuong
plt.scatter(dt.depth , dt.price)
plt.title("Sự tương quan giữa độ sâu và giá kim cương")
plt.xlabel("depth")
plt.ylabel("price")
plt.show()


#su tuong quan giua chiều dài va gia kim cuong
plt.scatter(dt.x , dt.price)
plt.title("Sự tương quan giữa chiều dài và giá kim cương")
plt.xlabel("x (length)")
plt.ylabel("price")
plt.show()
#su tuong quan giua chiều rộng va gia kim cuong
plt.scatter(dt.y , dt.price)
plt.title("Sự tương quan giữa chiều rộng và giá kim cương")
plt.xlabel("y (width)")
plt.ylabel("price")
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train,X_test,y_train,y_test=train_test_split(X,Xnhan,test_size=1/3.0,
random_state=5)


## hoi quy
import sklearn
from sklearn import linear_model
lm=linear_model.LinearRegression()
lm.fit(X_train,y_train)
y_pred=lm.predict(X_test)
err=mean_squared_error(y_test,y_pred)
print(round(err,3))
print(np.sqrt(err))

## Rung
print("=========rung============")
from sklearn.ensemble import RandomForestClassifier
tree=RandomForestClassifier(n_estimators=50, min_samples_leaf=5, max_depth=7,  random_state=0)
tree.fit(X_train,y_train)
y_pred=tree.predict(X_test)
err=mean_squared_error(y_test,y_pred)
err
np.sqrt(err)
#cay quyet dinh
print("=========cay============")
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
regressor = DecisionTreeRegressor(random_state= 0)
regressor.fit(X_train , y_train)
y_pred=regressor.predict(X_test)

err = mean_squared_error(y_test , y_pred)
print(err)
print(np.sqrt(err))

#tìm hàm hồi quy
Xprice=np.array(Xnhan.iloc[0:3]);
Xl=np.array(dt.x.iloc[0:3]);
def LR1(X,Y,eta,lanlap,theta0,theta1):
	m=len(X)
	print(m)
	for k in range (0,lanlap):
		print("lanlap:",k)
		for i in range(0,m):
			h_i=theta0+theta1*X[i]
			theta0 =theta0+eta*(Y[i]-h_i)*1
			print("phan tu ",i,"y=" ,Y[i],"h= ",h_i,"gia tri cua theta0= ",round(theta0,3))
			theta1 =theta1+eta*(Y[i]-h_i)*X[i]
			print("phan tu ",i,"gia tri cua theta1= ",round(theta1,3))
	return [round(theta0,3),round(theta1,3)]
theta =LR1(Xl,Xprice,0.001,2,0,1)
print(theta)