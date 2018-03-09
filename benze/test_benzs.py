
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
#print('Shape train: {}\n Shape test:{}'.format(train.shape, test.shape))

from sklearn.decomposition import PCA, FastICA
n_comp = 10  #选取10个

#PCA 主成分分析
pca = PCA(n_components= n_comp, random_state=42) #random_state指定后数据就不再变化
pca2_results_train = pca.fit_transform(train.drop(['y'],axis=1))
pca2_results_test = pca.transform(test)
#print(pca2_results_train)
#print(pca2_results_test)

#ICA 主成分分析
ica = FastICA(n_components = n_comp,random_state=42)
ica2_results_train = ica.fit_transform(train.drop(['y'], axis=1))
ica2_results_test = ica.fit_transform(test)

#Append decomosition components to datasets
for i in range(1, n_comp+1):
	train['pca_' + str(i)] = pca2_results_train[:,i-1]
	test['pca_' + str(i)] = pca2_results_test[:,i-1]

	train['ica_'+ str(i)] = ica2_results_train[:,i-1]
	test['ica_' +str(i)] = ica2_results_test[:, i-1]

print(train.columns)

y_train = train['y']
y_mean = np.mean(y_train)

x_train = train.drop('y', axis=1)
x_test = test 

#print('y_train',y_train)
#print('x_train',x_train)
#print('x_test',x_test)



###########################################模型效果验证方法###########
import warnings
warnings.filterwarnings('ignore')
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error



def rmse_cv(model, x_train, y_train):
	rmse = np.sqrt(-cross_val_score(model, x_train, y_train, scoring ='mean_squared_error', cv=3))
	return(rmse)

'''
#Lasso--------------------------------------

n=[0.01, 0.015, 0.02, 0.025, 0.03, 0.035]
cv_model = [rmse_cv(Lasso(alpha = num, max_iter =1000),x_train, y_train).mean() for num in n]
result = pd.Series(cv_model, index=n)
result.plot()
print(result.min())
plt.title('lasso with alphas')
plt.show()

#当alpha = 0.025时，损失最小
model=Lasso(alpha=0.025)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(pred)
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': pred})
output.to_csv('C:\\Users\\Administrator\\Desktop\\benz\\lasso.csv', index=False)
'''
'''
# XGBRegressor----------------------------------------------
num = [550,700,850,1000]
cv_model = [rmse_cv(XGBRegressor(max_depth=4, learning_rate=0.005, n_estimators=n,
	silent = True, objective='reg:linear', gamma=0, min_child_weight=1,subsample=1,
	colsample_bytree=1, base_score = y_mean, seed=0, missing =None),
	x_train, y_train).mean() for n in num]

result = pd.Series(cv_model,index = num)
result.plot()
print(result.min())
plt.title('XGBRegressor')
plt.show()
'''
'''
#选择n_estimators=850,进行模型计算
model = XGBRegressor(max_depth=4, learning_rate=0.005, n_estimators=850,
	silent = True, objective='reg:linear', subsample = 0.93, base_score=y_mean, seed=0,missing=None)
model.fit(x_train, y_train)
pred = model.predict(x_test)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y':pred})
output.to_csv('C:\\Users\\Administrator\\Desktop\\benz\\XGBRegressor.csv', index=False)
'''

############################# model ensemble 集成 ###############
'''
##1、bagging method
model = BaggingRegressor(base_estimator=XGBRegressor(max_depth=4, learning_rate=0.005, 
	n_estimators=800, silent=True, objective='reg:linear', subsample=0.95,base_score=y_mean,),
	n_estimators=10, max_samples=0.95, max_features=0.9)

model.fit(x_train, y_train)
pred = model.predict(x_test)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y':pred})
output.to_csv('C:\\Users\\Administrator\\Desktop\\benz\\bagging-XGBRegressor.csv', index=False)
'''
'''
##2、adaboost method 
model = AdaBoostRegressor(base_estimator= XGBRegressor(max_depth=4, learning_rate=0.005,
	n_estimators=800, silent=True, objective='reg:linear', subsample=0.95,base_score=y_mean,),
	n_estimators = 10, learning_rate=0.01, loss='linear', random_state=None)
model.fit(x_train, y_train)
pred = model.predict(x_test)

output = pd.DataFrame({'id':test['ID'].astype(np.int32), 'y':pred})
output.to_csv('C:\\Users\\Administrator\\Desktop\\benz\\adaboost-XGBRegressor.csv', index=False)
'''

########################Voting ensemble#########
#create sub models
estimators = []
estimators.append(('la', lasso(alpha=0.025)))
estimators.append(('XGBR', XGBRegressor(max_depth=4, learning_rate=0.005, n_estimators=850)))

#create the ensemble model
ensemble = VotingRegressor(estimators, voting='soft', weight=[4,6])
model = ensemble
model.fit(x_train, y_train)
pred = model.predict(x_test)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y':pred})
output.to_csv ('C:\\Users\\Administrator\\Desktop\\benz\\voting_ensemble.csv', index=False)

