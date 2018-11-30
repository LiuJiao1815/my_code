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
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder  #前处理，可以进行数据转码
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, chi2
from sklearn.decomposition import PCA, FastICA
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.grid_search import GridSearchCV

#########################导入数据##############################
train = pd.read_csv('C:\\Users\\Administrator\\Desktop\\benz\\train.csv')
test = pd.read_csv('C:\\Users\\Administrator\\Desktop\\benz\\test.csv')

############相关特征进行转码########################
for c in train.columns:
	if train[c].dtype == 'object':
		lbl = LabelEncoder()
		lbl.fit(list(train[c].values) + list(test[c].values))
		train[c] = lbl.transform(list(train[c].values))
		test[c] = lbl.transform(list(test[c].values))

#shape
print('train.shape:', train.shape)
print('test.shape:', test.shape)

##################################增加新的特征###############

#######通过PCA和ICA方法增加新特征
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

#-----------------------------------------------------

######确定训练数据和测试数据#########
y_train = train['y'].as_matrix()
#y_train.index = range(len(y_train))
y_mean = np.mean(y_train)

x_train = train.drop('y', axis=1)
x_train1 = x_train.iloc[:, 9 :377].as_matrix()
#x_train1.index = range(len(x_train1))

x_test = test 
x_test1 = x_test.iloc[:,9:377]
print(x_train.shape)
print(x_train.iloc[1, :])


###特殊的特征处理，使用 lasso 模型对高纬度的稀疏特征进行训练，输出的预测值作为新特征，取代原有的稀疏特征

#构建lasso模型
la =Lasso(alpha=0.025)
x_train_la = pd.DataFrame(np.zeros((len(x_train), 1)))
x_test_la = pd.DataFrame(np.zeros((len(x_test), 1)))


###########################################
la1 = la.fit(x_train1[0:2800], y_train[0:2800])
la2 = la.fit(x_train1[1400:4209], y_train[1400:4209])
la3 = la.fit(x_train1[list(range(0,1400))+list(range(2800, 4209))],  y_train[list(range(0,1400)) +list(range(2800, 4209))])
'''
la2.fit(x_train.iloc[0:2800, 45:80], y[0:2800])
la3.fit(x_train.iloc[0:2800, 80:115], y[0:2800])
la4.fit(x_train.iloc[0:2800, 115:150], y[0:2800])
la5.fit(x_train.iloc[0:2800, 150:185], y[0:2800])
la6.fit(x_train.iloc[0:2800, 185:220], y[0:2800])
la7.fit(x_train.iloc[0:2800, 220:265], y[0:2800])
la8.fit(x_train.iloc[0:2800, 265:300], y[0:2800])
la9.fit(x_train.iloc[0:2800, 300:335], y[0:2800])
la10.fit(x_train.iloc[0:2800, 335:377], y[0:2800])
'''

pred_la1=la1.predict(x_train1[2800:4209])
pred_la2= la2.predict(x_train1[0:1400])
pred_la3 = la3.predict(x_train1[1400:2800])

for i in range(1400):
	x_train_la.iloc[i+2800,:] = pred_la1[i]
	x_train_la.iloc[i, :] = pred_la2[i]
	x_train_la.iloc[i+1400,:] = pred_la3[i]


pred_la1 = la1.predict(x_test1)
pred_la2 = la2.predict(x_test1)
pred_la3 = la3.predict(x_test1)

for i in range(len(x_test1)):
	x_test_la.iloc[i] = (pred_la1[i] +pred_la2[i] +pred_la3[i])/3

x_train = pd.concat((x_train.iloc[:, 0:9], x_train.iloc[:,377:397], x_train_la), axis=1).as_matrix()
x_test = pd.concat((x_test.iloc[:, 0:9], x_test.iloc[:,377:397], x_test_la), axis=1).as_matrix()

print(x_train.shape)
print(x_test.shape)
#------------------------------------------------------------------------

###################stacking##################################
############################构建基模型#######################
#lasso模型
from sklearn.linear_model import LassoCV, Lasso
la = Lasso(alpha = 0.025, max_iter=100)

#---------------------------------------------------------
#SVR模型
from sklearn.svm import LinearSVR
svr = LinearSVR(
	C = 1,
	max_iter = 1000,
	)
#------------------------------------------------------------
#决策树回归模型
from sklearn.tree import DecisionTreeRegressor
dtc = DecisionTreeRegressor(
	criterion='mse', 
	max_depth=None,
	min_samples_split=2, 
	min_samples_leaf=1,
	max_features=None,
	random_state=None, 
	max_leaf_nodes=None,)
#------------------------------------------------------------
#随机森林回归模型
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(
	n_estimators = 100,
	max_depth = 5,
	min_samples_split = 2,
	min_samples_leaf = 1,
	min_weight_fraction_leaf=0.0,
	max_features = 1.0,
	min_impurity_split = 1e-7,
	bootstrap = True,
	oob_score = False,
	n_jobs = 1, 
	random_state = None,
	verbose = 0,
	warm_start = False)
#--------------------------------------------------------------
#K近邻回归模型
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(
	n_neighbors = 8,
	weights = 'uniform',
	algorithm = 'auto',
	leaf_size = 20,
	p = 2)
#-------------------------------------------------------------------
#DNN模型
#------------------------------------------------------------------

x_train_stacking = pd.DataFrame(np.zeros((len(x_train), 5)))
x_test_stacking = pd.DataFrame(np.zeros((len(x_test), 5)))

###########################fold-1################

la.fit(x_train[0:2800], y_train[0:2800])
svr.fit(x_train[0:2800], y_train[0:2800])
dtc.fit(x_train[0:2800], y_train[0:2800])
rf.fit(x_train[0:2800], y_train[0:2800])
knn.fit(x_train[0:2800], y_train[0:2800])
#------------------------------------------------------

pred_la=la.predict(x_train[2800:4209])
pred_svr = svr.predict(x_train[2800:4209])
pred_dtc=dtc.predict(x_train[2800:4209])
pred_rf = rf.predict(x_train[2800:4209])
pred_knn = knn.predict(x_train[2800:4209])

for i in range(1409):
	x_train_stacking.iloc[2800+i,0] = pred_la[i]
	x_train_stacking.iloc[2800+i,1] = pred_svr[i]
	x_train_stacking.iloc[2800+i,2] = pred_dtc[i]
	x_train_stacking.iloc[2800+i,3] = pred_rf[i]
	x_train_stacking.iloc[2800+i,4] = pred_knn[i]

pred_la=la.predict(x_test)
pred_svr = svr.predict(x_test)
pred_dtc=dtc.predict(x_test)
pred_rf = rf.predict(x_test)
pred_knn = knn.predict(x_test)

for i in range(len(x_test)):
	x_test_stacking.iloc[i,0] = x_test_stacking.iloc[i,0] +pred_la[i]
	x_test_stacking.iloc[i,1] = x_test_stacking.iloc[i,1] + pred_svr[i]
	x_test_stacking.iloc[i,2] = x_test_stacking.iloc[i,2] + pred_dtc[i]
	x_test_stacking.iloc[i,3] = x_test_stacking.iloc[i,3] + pred_rf[i]
	x_test_stacking.iloc[i,4] = x_test_stacking.iloc[i,4] + pred_knn[i]

#------------------------------------------------

############################fold-2##############################
la.fit(x_train[1400:4209], y_train[1400:4209])
svr.fit(x_train[1400:4209], y_train[1400:4209])
dtc.fit(x_train[1400:4209], y_train[1400:4209])
rf.fit(x_train[1400:4209], y_train[1400:4209])
knn.fit(x_train[1400:4209], y_train[1400:4209])
#------------------------------------------------------

pred_la=la.predict(x_train[0:1400])
pred_svr = svr.predict(x_train[0:1400])
pred_dtc=dtc.predict(x_train[0:1400])
pred_rf = rf.predict(x_train[0:1400])
pred_knn = knn.predict(x_train[0:1400])

for i in range(1400):
	x_train_stacking.iloc[i,0] = pred_la[i]
	x_train_stacking.iloc[i,1] = pred_svr[i]
	x_train_stacking.iloc[i,2] = pred_dtc[i]
	x_train_stacking.iloc[i,3] = pred_rf[i]
	x_train_stacking.iloc[i,4] = pred_knn[i]

pred_la=la.predict(x_test)
pred_svr = svr.predict(x_test)
pred_dtc=dtc.predict(x_test)
pred_rf = rf.predict(x_test)
pred_knn = knn.predict(x_test)

for i in range(len(x_test)):
	x_test_stacking.iloc[i,0] = x_test_stacking.iloc[i,0] +pred_la[i]
	x_test_stacking.iloc[i,1] = x_test_stacking.iloc[i,1] + pred_svr[i]
	x_test_stacking.iloc[i,2] = x_test_stacking.iloc[i,2] + pred_dtc[i]
	x_test_stacking.iloc[i,3] = x_test_stacking.iloc[i,3] + pred_rf[i]
	x_test_stacking.iloc[i,4] = x_test_stacking.iloc[i,4] + pred_knn[i]

#---------------------------------------------------------------

############################fold-3##############################
la.fit(x_train[list(range(0,1400))+list(range(2800, 4209))],  y_train[list(range(0,1400)) +list(range(2800, 4209))])
svr.fit(x_train[list(range(0,1400))+list(range(2800, 4209))],  y_train[list(range(0,1400)) +list(range(2800, 4209))])
dtc.fit(x_train[list(range(0,1400))+list(range(2800, 4209))],  y_train[list(range(0,1400)) +list(range(2800, 4209))])
rf.fit(x_train[list(range(0,1400))+list(range(2800, 4209))],  y_train[list(range(0,1400)) +list(range(2800, 4209))])
knn.fit(x_train[list(range(0,1400))+list(range(2800, 4209))],  y_train[list(range(0,1400)) +list(range(2800, 4209))])
#------------------------------------------------------

pred_la=la.predict(x_train[1400:2800])
pred_svr = svr.predict(x_train[1400:2800])
pred_dtc=dtc.predict(x_train[1400:2800])
pred_rf = rf.predict(x_train[1400:2800])
pred_knn = knn.predict(x_train[1400:2800])

for i in range(1400):
	x_train_stacking.iloc[1400+i,0] = pred_la[i]
	x_train_stacking.iloc[1400+i,1] = pred_svr[i]
	x_train_stacking.iloc[1400+i,2] = pred_dtc[i]
	x_train_stacking.iloc[1400+i,3] = pred_rf[i]
	x_train_stacking.iloc[1400+i,4] = pred_knn[i]

pred_la=la.predict(x_test)
pred_svr = svr.predict(x_test)
pred_dtc=dtc.predict(x_test)
pred_rf = rf.predict(x_test)
pred_knn = knn.predict(x_test)

for i in range(len(x_test)):
	x_test_stacking.iloc[i,0] = x_test_stacking.iloc[i,0] +pred_la[i]
	x_test_stacking.iloc[i,1] = x_test_stacking.iloc[i,1] + pred_svr[i]
	x_test_stacking.iloc[i,2] = x_test_stacking.iloc[i,2] + pred_dtc[i]
	x_test_stacking.iloc[i,3] = x_test_stacking.iloc[i,3] + pred_rf[i]
	x_test_stacking.iloc[i,4] = x_test_stacking.iloc[i,4] + pred_knn[i]
#------------------------------------------------------------------
#####################对测试集结果进行平均化处理#####33
print(x_test_stacking)

for i in range(len(x_test)):
	for j in range(5):
		x_test_stacking.iloc[i,j] = x_test_stacking.iloc[i, j]/3
		
print(x_test_stacking)


###################################第二层用xgboost#############
xgb = XGBRegressor(
	max_depth=4, 
	learning_rate=0.005, 
	n_estimators=500,
	silent = True, 
	objective='reg:linear', 
	subsample = 0.93, 
	base_score=y_mean, 
	seed=0,
	missing=None)
xgb.fit(x_train_stacking, y_train)
pred = xgb.predict(x_test_stacking)
print(pred)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y':pred})
output.to_csv('C:\\Users\\Administrator\\Desktop\\benz\\new\\test_stacking.csv', index=False)



