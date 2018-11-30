#benze kaggle 比赛分析	
#以奔驰新车系统安全测试历史数据，建立生新车通过系统测试所需时间的预测模型。

#benzs 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder  #前处理，可以进行数据转码
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import RidgeCV, Ridge 
import xgboost as xgb 
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
import matplotlib.pyplot as plt 

#导入数据
train = pd.read_csv('C:\\Users\\Administrator\\Desktop\\benz\\train.csv')
test = pd.read_csv('C:\\Users\\Administrator\\Desktop\\benz\\test.csv')

#转码
for c in train.columns:
	if train[c].dtype == 'object':
		lbl = LabelEncoder()
		lbl.fit(list(train[c].values) + list(test[c].values))
		train[c] = lbl.transform(list(train[c].values))
		test[c] = lbl.transform(list(test[c].values))

#shape
print('train.shape:', train.shape)
print('test.shape:', test.shape)


#通过PCA和ICA方法增加新特征
from sklearn.decomposition import PCA, FastICA
n_comp = 10  #选取10个

#PCA 主成分分析
pca = PCA(n_components= n_comp, random_state=42) #random_state指定后数据就不再变化
pca2_results_train = pca.fit_transform(train.drop(['y'],axis=1))
pca2_results_test = pca.transform(test)
#print(pca2_results_train)
#print(pca2_results_test)

#ICA 独立成分分析
ica = FastICA(n_components = n_comp,random_state=42)
ica2_results_train = ica.fit_transform(train.drop(['y'], axis=1))
ica2_results_test = ica.fit_transform(test)

#把新特征添加到原始特征中
for i in range(1, n_comp+1):
	train['pca_' + str(i)] = pca2_results_train[:,i-1]
	test['pca_' + str(i)] = pca2_results_test[:,i-1]

	train['ica_'+ str(i)] = ica2_results_train[:,i-1]
	test['ica_' +str(i)] = ica2_results_test[:, i-1]

print(train.columns)

#确定训练数据和测试数据
y_train = train['y']
y_mean = np.mean(y_train)

x_train = train.drop('y', axis=1)
x_test = test 


###########################################交叉验证确定模型的最优参数###########
import warnings
warnings.filterwarnings('ignore')
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

def rmse_cv(model, x_train, y_train):
	rmse = np.sqrt(-cross_val_score(model, x_train, y_train, scoring ='mean_squared_error', cv=3))
	return(rmse)

'''
#Lasso模型参数确定--------------------------------------

n=[0.01, 0.015, 0.02, 0.025, 0.03, 0.035]
cv_model = [rmse_cv(Lasso(alpha = num, max_iter =1000),x_train, y_train).mean() for num in n]
result = pd.Series(cv_model, index=n)
result.plot()
print(result.min())
plt.title('lasso with alphas')
plt.show()
#当alpha = 0.025时，损失最小


########################################Lasso模型预测###############

model=Lasso(alpha=0.025)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(pred)
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': pred})
output.to_csv('C:\\Users\\Administrator\\Desktop\\benz\\new\\lasso.csv', index=False)
'''

'''
# XGBRegressor模型参数确定----------------------------------------------
n = [400,500,550,700,850]
cv_model = [rmse_cv(XGBRegressor(max_depth=4, learning_rate=0.005, n_estimators=num,
	silent = True, objective='reg:linear', gamma=0, min_child_weight=1,subsample=1,
	colsample_bytree=1, base_score = y_mean, seed=0, missing =None),
	x_train, y_train).mean() for num in n]

result = pd.Series(cv_model,index = n)
result.plot()
print(result.min())
plt.title('XGBRegressor')
plt.show()
#当n_estimators=500时误差最小


#########################选择n_estimators=500,进行XGBRegressor模型计算########

model = XGBRegressor(max_depth=4, learning_rate=0.005, n_estimators=500,
	silent = True, objective='reg:linear', subsample = 0.93, base_score=y_mean, seed=0,missing=None)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(pred)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y':pred})
output.to_csv('C:\\Users\\Administrator\\Desktop\\benz\\new\\XGBRegressor.csv', index=False)
'''

'''
#################################融合模型 model ensemble#############
#1、bagging method 并行算法#########################

model = BaggingRegressor(
	base_estimator=XGBRegressor(max_depth=4, learning_rate=0.005, n_estimators=500,
		silent=True, objective="reg:linear", subsample=0.95, base_score=y_mean),
	n_estimators=10, max_samples=0.95, max_features=0.9)

model.fit(x_train, y_train)
pred = model.predict(x_test)

output = pd.DataFrame({'id': test['ID'],  'y':pred})
output.to_csv('C:\\Users\\Administrator\\Desktop\\benz\\new\\bagging-XGBRegressor.csv', index=False)
'''

##2、adaboost method 串行算法------------------------------------

model = AdaBoostRegressor(
	base_estimator=XGBRegressor(max_depth=4, learning_rate=0.005, n_estimators=500,
		silent=True, objective="reg:linear", subsample=0.95, base_score=y_mean),
	n_estimators=10, learning_rate=0.01)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(pred)
#output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': pred})
#output.to_csv('C:\\Users\\Administrator\\Desktop\\benz\\new\\adaboost-XGBRegressor.csv', index=False)

'''
################DNN模型###############

from keras.model import Sequential
from keras.layers.core import Dense, Activation, Dropout

model = Sequential()
model.add(Dense(300, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='relu')) 

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train.as_matrix(), y_train.as_matrix(), nb_epoch=10, batch_size=30)

y_pred = model.predict(x_test.as_matrix().reshape(len(x_test)))
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv()

'''