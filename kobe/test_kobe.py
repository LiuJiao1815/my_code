#kobe shuting

import pandas as pd 
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import math
import matplotlib.pyplot as plt 
import seaborn as sns 
import xgboost as xgb 
from xgboost.sklearn import XGBClassifier
from sklearn.feature_selection import VarianceThreshold, RFE,SelectKBest, chi2
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA, KernelPCA 
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, AdaBoostClassifier



'''
inputfile = 'F:\\kaggle_kobe\\data.csv'
data = pd.read_csv(inputfile)

#统计action_type的种类
action_type = []
for i in range(len(data)):
	if data['action_type'][i] not in action_type:
		action_type.append(data['action_type'][i])
print(action_type)
print('kobe 的得分方式种类数量：',len(action_type))

#统计combine_action_type 的种类
combine_action_type= []
for i in range(len(data)):
	if data['combined_shot_type'][i] not in combine_action_type:
		combine_action_type.append(data['combined_shot_type'][i])
print(combine_action_type)
print('kobe 的combine的得分方式的数量', len(combine_action_type))

#统计kobe在每一种得分方式下的成功率

shot_try_count = {}    #存放各种得分方式的出手次数
shot_made_count = {}    #存放各种得分方式的出手并且成功的次数
shot_type_success = {} #存放各种得分方式的出手成功概率
shot_type_label = {}   #存放各种得分方式标记
#初始化
for type in action_type:
	shot_try_count[type] = 0
for type in action_type:
	shot_made_count[type] = 0

for i in range(len(data)):
	if data['shot_made_flag'][i]>=0:    #注意很巧妙的一种做法，有效的忽略了空值
		shot_try_count[data['action_type'][i]] =shot_try_count[data['action_type'][i]] +1
		if data['shot_made_flag'][i] ==1:
			shot_made_count[data['action_type'][i]]= shot_made_count[data['action_type'][i]] +1

for type in action_type:
	print('***',type)
	if shot_try_count[type]>0:
		shot_type_success[type] = shot_made_count[type]/shot_try_count[type]
	else:
		shot_type_success[type] = 0 #训练集中没有出现过的得分动作的成功率为0

print(shot_type_success)


for type in action_type:
	if shot_type_success[type]>0.8:
		shot_type_label[type] = 7

	elif shot_type_success[type]>0.7:
		shot_type_label[type] = 6

	elif shot_type_success[type]>0.6:
		shot_type_label[type] =5

	elif shot_type_success[type]>0.5:
		shot_type_label[type] =4

	elif shot_type_success[type]>0.4:
		shot_type_label[type] = 3

	elif shot_type_success[type]>0.3:
		shot_type_label[type] =2
	elif shot_type_success[type]>0.2:
		shot_type_label[type] =1

	else:
		shot_type_label[type] =0
print(shot_type_label)

#将data中的action_type 变换过来
for i in range(len(data)):
	data['action_type'][i] =shot_type_label[data['action_type'][i]]
	
print(data['action_type'])
data.to_excel('F:\\kaggle_kobe\\data_01.xls')   #保存将action_type特征转换为0-7数值的数据

############################################################
#对出手位置的相关数据进行预处理

data = pd.read_excel('F:\\kaggle_kobe\\data_01.xls')

#loc_x数据标准化处理
data['loc_x'] = (data['loc_x']-data['loc_x'].mean())/data['loc_x'].std()
print(data['loc_x'])

#loc_y 数据标准化处理
data['loc_y'] = (data['loc_y']-data['loc_y'].mean())/data['loc_y'].std()
print(data['loc_y'])

#shot_distance
print(data['shot_distance'].describe())
data['shot_distance'] = (data['shot_distance']-data['shot_distance'].mean())/data['shot_distance'].std()
print(data['shot_distance'])

#shot_zone_area
print(data['shot_zone_area'])
shot_zone_area=[]
for i in range(len(data)):
	if data['shot_zone_area'][i] =='Left Side(L)':
		data['shot_zone_area'][i] =0

	elif data['shot_zone_area'][i] =='Left Side Center(LC)':
		data['shot_zone_area'][i] =1

	elif data['shot_zone_area'][i] == 'Right Side(R)':
		data['shot_zone_area'][i] = 2

	elif data['shot_zone_area'][i] == 'Right Side Center':
		data['shot_zone_area'][i] =3

	elif data['shot_zone_area'][i] == 'Center':
		data['shot_zone_area'][i] = 4

	else:
		data['shot_zone_area'][i] =5
print(data['shot_zone_area'])

data.to_excel('F:\\kaggle_kobe\\data_02.xls')  #保存转换过loc_X, loc_y, shot_distance, shot_zone_area 的数据
'''

###########################################################
#对时间相关的特征进行预处理
data = pd.read_excel('F:\\kaggle_kobe\\data_02.xls')
print(data.columns)
print(data['game_date'][0])

#gamedata 特征的处理
for i in range(len(data)):
	data['game_date'][i] = (int(data['game_date'][i][:4])-1996)*12+int(data['game_date'][i][5:7])-11+1
	print(data['game_date'][i])

#比赛结束特征的处理，特征格式比较合理，暂时不做处理
#每一节比赛剩余分钟数minutes_remaining特征的处理，特征格式比较合理，暂时不做处理

############2分球，3分球特征shot_type 的预处理#########

print(data['shot_type'])
for i in range(len(data)):
	data['shot_type'][i] =int(data['shot_type'][i][0])   #取出第一个数字
	print(i)
data.to_excel('F:\\kaggle_kobe\\data_03.xls')


'''
###################处理combined shot type#################################
data = pd.read_excel('F:\\kaggle_kobe\\data_03.xls')
print(data.iloc[0,:])

t = []
for i in range(len(data)):
	if data['combined_shot_type'][i] not in t:
		t.append(data['combined_shot_type'][i])
print(t)

for i in range(len(data)):
	data['combined_shot_type'][i] = t.index(data['combined_shot_type'][i])
print(data['combined_shot_type'])


############处理shot_zone_basic#############
b =[]
for i in range(len(data)):
	if data['shot_zone_basic'][i] =='Restricted Area':
		data['shot_zone_basic'][i] =1

	elif data['shot_zone_basic'][i] =='Mid-Range':
		data['shot_zone_basic'][i] =2

	elif data['shot_zone_basic'][i] =='In The Paint(Non-RA)':
		data['shot_zone_basic'] =3

	elif data['shot_zone_basic'][i] == 'Above the Break 3':
		data['shot_zone_basic'] =4

	elif data['shot_zone_basic'][i] =='Right Corner 3':
		data['shot_zone_basic'][i] =5

	elif data['shot_zone_basic'][i] =='Left Corner 3':
		data['shot_one_basic'][i] =6
	else:
		data['shot_zone_basic'][i] =7


#############处理shot_zone_range###################

for i in range(len(data)):
	if data['shot_zone_range'][i] =='Less Than 8 ft.':
		data['shot_zone_range'][i] = 1
	elif data['shot_zone_range'][i] == '8-16 ft.':
		data['shot_zone_range'][i] = 2
	elif data['shot_zone_range'][i] == '16-24 ft.':
		data['shot_zone_range'][i] =3
	elif data['shot_zone_range'][i] =='24+ft.':
		data['shot_zone_range'][i] =4
	else:
		data['shot_zone_range'][i]=5

data.to_excel('F:\\kaggle_kobe\\data_04.xls')
'''
'''
########################opponent对手特征处理#############
data =pd.read_excel('F:\\kaggle_kobe\\data_04.xls')

t = []
for i in range(len(data)):
	if data['opponent'][i] not in t :
		t.append(data['opponent'][i])

print(t)
for i in range(len(data)):
	data['opponent'][i] = t.index(data['opponent'][i])+1
	#一种把特征值转换为数字的方式
data.to_excel('F:\\kaggle_kobe\\data_05.xls')

'''
'''
################lat特征处理############

data = pd.read_excel('F:\\kaggle_kobe\\data_05.xls')

data[['lat','loc_x','loc_y','lon', 'shot_distance','minutes_remaining','seconds_remaining']] =(data[['lat','loc_x','loc_y','lon', 'shot_distance','minutes_remaining','seconds_remaining']]-data[['lat','loc_x','loc_y','lon', 'shot_distance','minutes_remaining','seconds_remaining']].min())/(data[['lat','loc_x','loc_y','lon', 'shot_distance','minutes_remaining','seconds_remaining']].max()-data[['lat','loc_x','loc_y','lon', 'shot_distance','minutes_remaining','seconds_remaining']].min())
print(data.iloc[0,:])
data.to_excel('F:\\kaggle_kobe\\data_06.xls')
'''
'''
####################game data特征处理##############
data = pd.read_excel('F:\\kaggle_kobe\\data_06.xls')
data['month']=data['shot_id']
data['year'] = data['shot_id']

for i in range(len(data)):

	data['year'][i] = int(data['game_date1'][i][:4])-1995
	data['month'][i] = int(data['game_date1'][i][5:7])
print(data['month'])
data.to_excel('F:\\kaggle_kobe\\data_07.xls')
'''

'''

############################lat特征离散化##########
data=pd.read_excel('F:\\kaggle_kobe\\data_07.xls')

k=20
d = pd.cut(data['lat'],k,labels=range(k))   #等宽离散化，各个类比依次命名为0,1,2,3，
data['lat'] =d

################loc_x, loc_y 离散化
k=20 
d = pd.cut(data['loc_x'], k ,labels =range(k)) #等宽离散化，各个类比依次命名为0,1,2,3
data['loc_x']=d

d= pd.cut(data['loc_y'], k ,labels=range(k))   #等宽离散化，各个类比依次为0,1,2,3，
data['loc_y'] =d

##################lon minutes, seconds, shot distance 离散化#########
k =20
d=pd.cut(data['lon'], k,labels=range(k))   #等宽离散化，各个类别依次命名为0,1,2,3，
data['lon']=d

k=20
d=pd.cut(data['minutes_remaining'], k , labels=range(k))  #等宽离散化
data['minutes_remaining']=d

k=20
d=pd.cut(data['seconds_remaining'], k, labels=range(k))   #等宽离散化
data['seconds_remaining'] = d

k = 20
d = pd.cut(data['shot_distance'], k , labels=range(k))     #等宽离散化
data['shot_distance'] = d

print(data.iloc[0,:])
data.to_excel('F:\\kaggle_kobe\\data_08.xls')
'''
'''
###################数据可视化#############
data=pd.read_excel('F:\\kaggle_kobe\\data_08.xls')

#投篮中与不中的分布
ax = plt.axes()
sns.countplot(x='shot_made_flag',data=data, ax=ax)
ax.set_title('Target class distribution')
plt.show()

#各个特征对于shotmadeflag的箱型图
f ,axarr = plt.subplots(4,2,figsize=(18,18))
sns.boxplot(x='lat',y='shot_made_flag', data=data, showmeans=True, ax=axarr[0,0])
sns.boxplot(x='lon',y='shot_made_flag',data=data,showmeans=True,ax=axarr[0,1])
sns.boxplot(x='loc_y', y='shot_made_flag', data=data, showmeans=True, ax=axarr[1,0])
sns.boxplot(x='loc_x',y='shot_made_flag', data=data,showmeans=True,ax=axarr[1,1])
sns.boxplot(x='minutes_remaining', y='shot_made_flag', data=data, showmeans=True, ax=axarr[2,0])
sns.boxplot(x='seconds_remaining',y='shot_made_flag', data=data, showmeans=True,ax=axarr[2,1])
sns.boxplot(x='shot_distance', y='shot_made_flag', data=data,showmeans=True,ax=axarr[3,0])

axarr[0,0].set_title('Latitude')   #纬度
axarr[0,1].set_title('Longitude')  #经度
axarr[1,0].set_title('loc_y')
axarr[1,1].set_title('loc_x')
axarr[2,0].set_title('Minutes seconds_remaining')
axarr[2,1].set_title('Seconds remaining')
axarr[3,0].set_title('Shot distance')
plt.tight_layout()
plt.show()


#各个特征对于投篮是否命中的分布图
f , axarr =plt.subplots(8, figsize=(15,25))
sns.countplot(x='combined_shot_type', hue='shot_made_flag', data=data, ax =axarr[0])
sns.countplot(x='playoffs', hue='shot_made_flag', data=data, ax=axarr[1])
sns.countplot(x='shot_type',hue='shot_made_flag', data=data,ax=axarr[2])
sns.countplot(x='shot_zone_area', hue='shot_made_flag', data=data, ax=axarr[3])
sns.countplot(x='shot_zone_basic', hue='shot_made_flag',data=data, ax=axarr[4])
sns.countplot(x='shot_zone_range',hue='shot_made_flag', data=data,ax=axarr[5])
sns.countplot(x='period', hue='shot_made_flag', data=data, ax=axarr[6])

#axarr[0].set_title('Combined shot type')
#axarr[1].set_title('playoffs')
#axarr[2].set_title('shot_type')
#axarr[3].set_title('shot_zone_area')
#axarr[4].set_title('shot_zone_basic')
#axarr[5].set_title('shot_zone_range')
#axarr[6].set_title('period')

plt.tight_layout()
plt.show()
'''
'''
###########创建新的特征#############
data=pd.read_excel('F:\\kaggle_kobe\\data_08.xls')

#创建一个新的特征：每阶段的最后五秒
data['seconds_from_period_end'] = 60*data['minutes_remaining']+data['seconds_remaining']
#布尔运算，大于5秒的记为0，小5秒的记为1
data['last_5_sec_in_period_end']= (data['seconds_from_period_end']<5).astype('int')
data.drop('seconds_from_period_end',axis=1, inplace=True)
print(data['last_5_sec_in_period_end'])
data.to_excel('F:\\kaggle_kobe\\data_08.xls')


#创建一个新特征，主场-home_play
#下面相当于一个布尔运算，当字符串中包含VS时为1，不包含时为0
data['home_play'] = data['matchup'].str.contains('vs').astype('int')
print(data['home_play'])
data.drop('matchup',axis=1,inplace=True)
print(data.iloc[0,:])
data.to_excel('F:\\kaggle_kobe\\data_08.xls')
'''

'''
####################one-hot转码###############
data = pd.read_excel('F:\\kaggle_kobe\\data_08.xls')
print(data.columns)
categorical_cols = ['action_type', 'combined_shot_type', 'lat', 'loc_x', 'loc_y',
	'lon', 'period', 'playoffs','shot_distance', 'shot_type', 'shot_zone_area',
	'shot_zone_basic', 'shot_zone_range', 'opponent', 'month', 'year', 'home_play',
	'last_5_sec_in_period_end']

for cc in categorical_cols:
	dummies = pd.get_dummies(data[cc])
	dummies = dummies.add_prefix('{}#'.format(cc))
	data.drop(cc, axis=1, inplace=True)
	data = data.join(dummies)

print(data.iloc[0,:])
data.to_excel('F:\\kaggle_kobe\\data_09_feature01.xls')
'''

'''
###########特征选择，减少特征的个数############
data = pd.read_excel('F:\\kaggle_kobe\\data_09_feature01.xls')
#提取出flag为空的记录
unknown_mask = data['shot_made_flag'].isnull()
#print(unknown_mask)

#提取出训练输入x, 训练目标Y, 测试输入Datasubmit
datasubmit = data[unknown_mask]    #仔细分析这一步操作，选出了shot_made_flag为空值的所有项
datasubmit.drop('shot_made_flag', axis=1, inplace=True)
datasubmit.drop('shot_id', axis=1,inplace=True)
datasubmit.drop('game_date1',axis=1,inplace=True)
datasubmit.drop('game_event_id', axis=1, inplace=True)
datasubmit.drop('game_id', axis=1, inplace=True)
datasubmit.drop('season', axis=1, inplace=True)
datasubmit.drop('team_id',axis=1, inplace=True)
datasubmit.drop('team_name', axis=1, inplace=True)

data_train=data[~unknown_mask]   #与unknown_mask的操作是相反的
y = data_train['shot_made_flag'] #选择训练的y值
data_train.drop('shot_made_flag', axis=1, inplace=True)
data_train.drop('shot_id', axis=1, inplace=True)
data_train.drop('game_date1',axis=1,inplace=True)
data_train.drop('game_event_id', axis=1, inplace=True)
data_train.drop('game_id', axis=1, inplace=True)
data_train.drop('season', axis=1, inplace=True)
data_train.drop('team_id',axis=1, inplace=True)
data_train.drop('team_name', axis=1, inplace=True)
x=data_train
print(x.columns)
##################################################################
#Variance Threshold select feature##################

threshold = 0.90
vt = VarianceThreshold().fit(x)

#Find feature names
feat_var_threshold = x.columns[vt.variances_>threshold*(1-threshold)]
#print(feat_var_threshold)
#print(len(feat_var_threshold))

#Top 20 most important features using RF model 

model =RandomForestClassifier()
model.fit(x,y)

feature_imp = pd.DataFrame(model.feature_importances_, index=x.columns, columns=['importance'])
#print(feature_imp)
feat_imp_20 = feature_imp.sort_values('importance', ascending=False).head(20).index
#ascending=False 表示按降序排列
#print(feat_imp_20)


#Univarite feature selection 单变量特征选择##########

x_minmax =MinMaxScaler(feature_range=(0,1)).fit_transform(x)
x_scored =SelectKBest(score_func=chi2, k='all').fit(x_minmax, y)
feature_scoring =pd.DataFrame({
	'feature': x.columns,
	'score':x_scored.scores_
	})

feat_scored_20 = feature_scoring.sort_values('score', ascending=False).head(20)['feature'].values
#print(feat_scored_20)


#Recursive feature elimination  循环特征淘汰

rfe =RFE(LogisticRegression(),20)
rfe.fit(x,y)

feature_rfe_scoring = pd.DataFrame({
	'feature':x.columns,
	'score': rfe.ranking_
	})

feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score']==1]['feature'].values
#print('feat_rfe_20:', feat_rfe_20)


################################
features = np.hstack([feat_var_threshold,feat_imp_20, feat_scored_20, feat_rfe_20])
features = np.unique(features)   #去重
#print(features)


features = ['action_type#1','action_type#2', 'action_type#4', 'action_type#6',
 'action_type#7', 'combined_shot_type#0' ,'combined_shot_type#1',
 'combined_shot_type#2' ,'game_date', 'home_play#0', 'home_play#1',
 'last_5_sec_in_period_end#1', 'lat#13', 'lat#14', 'lat#15' ,'lat#16', 'lat#17',
 'lat#18', 'lat#3', 'lat#4', 'lat#5' ,'lat#6', 'lat#9', 'loc_x#10' ,'loc_y#1',
 'loc_y#10', 'loc_y#12', 'loc_y#13', 'loc_y#2', 'loc_y#3', 'loc_y#4', 'loc_y#5',
 'loc_y#6', 'lon#10' ,'minutes_remaining', 'month#1', 'month#11', 'month#12',
 'month#2', 'month#3', 'month#4' ,'period#1' ,'period#2', 'period#3', 'period#4',
 'playoffs#0' ,'playoffs#1', 'seconds_remaining' ,'shot_distance#0',
 'shot_distance#11' ,'shot_distance#3' ,'shot_distance#4' ,'shot_distance#5',
 'shot_distance#6', 'shot_distance#8' ,'shot_distance#9' ,'shot_type#2',
 'shot_type#3' ,'shot_zone_area#0', 'shot_zone_area#1','shot_zone_area#2',
 'shot_zone_area#5', 'shot_zone_range#1', 'shot_zone_range#2',
 'shot_zone_range#3' ,'shot_zone_range#5']

x= x[features]
datasubmit = datasubmit[features]
print(datasubmit)

x.to_excel('F:\\kaggle_kobe\\data_09_x_feature01.xls')
pd.DataFrame(y).to_excel('F:\\kaggle_kobe\\data_09_y_feature01.xls')
datasubmit.to_excel('F:\\kaggle_kobe\\data_09_datasubmit_feature01.xls')
'''

'''
#本分析中并没有选用pca的分析结果
###############feature PCA ############
x =pd.read_excel('F:\\kaggle_kobe\\data_09_x_feature01.xls')
y =pd.read_excel('F:\\kaggle_kobe\\data_09_y_feature01.xls')
datasubmit = pd.read_excel('F:\\kaggle_kobe\\data_09_datasubmit_feature01.xls')

components=15
pca = PCA(n_components = components)
x = pca.fit_transform(x)
pca_variance_explained_df =pd.DataFrame({
	'component': np.arange(1, components+1),
	'variance_explained':pca.explained_variance_ratio_  #每个特征所占的比例
	})

ax =sns.barplot(x ='component', y='variance_explained', data =pca_variance_explained_df)
ax.set_title('PCA-Variance explained')
plt.show()
print(x)
'''
####################################################
#################Evaluate Algorithms###############
warnings.filterwarnings('ignore')
x_train =pd.read_excel('F:\\kaggle_kobe\\data_09_x_feature01.xls')
y_train =pd.read_excel('F:\\kaggle_kobe\\data_09_y_feature01.xls')
x_test = pd.read_excel('F:\\kaggle_kobe\\data_09_datasubmit_feature01.xls')

#for cv 交叉验证--------
seed = 7
processors =1
num_folds = 5
num_instances =len(x_train)
scoring = 'log_loss'
kfold = KFold(n = num_instances, n_folds=num_folds,random_state = seed)
#-----------------------------

'''
#####Bagged DTC CV for evaluate Algoritithms########
cart = DecisionTreeClassifier()
num_trees =100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, x , y,cv=kfold, scoring=scoring, n_jobs=processors)
print('Bagged DTC CV for evaluate Algorithms:', '({0:.3f})+/- ({1:.3f})'.format(results.mean(),results.std()))

##############Random Forest cv for evaluate Algorithms########
num_trees = 100
num_features =10
model =RandomForestClassifier(n_estimators = num_trees, max_features=num_features)
results =cross_val_score(model, x,y, cv=kfold, scoring =scoring, n_jobs=processors)
print('RF CV for evaluate Algorithms:', '({0:.3f}) +/- ({1:.3f})'.format(results.mean(),results.std()))


#####Extra Trees CV for evaluate Algorithms###########
num_trees = 100
num_features =10
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=num_features)
results = cross_val_score(model, x, y, cv=kfold, scoring=scoring, n_jobs=processors)
print('Extra Trees CV for evaluate Algorithms:', '({0:.3f}) +/- ({1:.3f})'.format(results.mean(),results.std()))


#####LR CV for evaluate Algorithms########
penalty = 'l2'
C =10
max_iter=100
model = LogisticRegression(penalty='l2', C=10, max_iter=100)
results = cross_val_score(model, x, y, cv=kfold, scoring=scoring, n_jobs=processors)
print('LR CV for evaluate Algorithms:', '({0:.3f})+/-({1:.3f})'.format(results.mean(),results.std()))


#########Adaboost CV for evalute Algorithms########
model = AdaBoostClassifier(n_estimators=100, random_state=seed)
results = cross_val_score(model, x, y, cv=kfold, scoring=scoring, n_jobs=processors)
print('Adaboost CV for evaluate Algorithms:', '({0:.3f})+/-({1:.3f})'.format(results.mean(), results.std()))

#######GBDT CV for evalate Algorithms ############
model = GradientBoostingClassifier(n_estimators=100, random_state=seed)
results = cross_val_score(model, x, y,cv=kfold, scoring=scoring, n_jobs=processors)
print('GBDT CV for evaluate Algorithms:', '({0:.3f})+/-({1:.3f})'.format(results.mean(), results.std()))
'''

###################Hyperparameter tuning 超参调节##########
'''
Bagged DTC CV for evaluate Algorithms: (-0.640)+/- (0.006)
RF CV for evaluate Algorithms: (-0.639) +/- (0.010)
Extra Trees CV for evaluate Algorithms: (-0.718) +/- (0.019)
LR CV for evaluate Algorithms: (-0.610)+/-(0.005)
Adaboost CV for evaluate Algorithms: (-0.691)+/-(0.000)
GBDT CV for evaluate Algorithms: (-0.606)+/-(0.004)

'''

'''
##Logistic Regression GridSearchCV ###########

lr_grid = GridSearchCV(
	estimator = LogisticRegression(random_state = seed),
	param_grid ={
		'penalty':['l1', 'l2'],
		'C': [0.001, 0.01, 1, 10, 100,1000]
	},
	cv= kfold,
	scoring = scoring,
	n_jobs = processors)
lr_grid.fit(x,y)
print(lr_grid.best_score_)
print(lr_grid.best_params_)

#LR, BEST:{'penalty':l2, 'C':1000}
'''

'''
##RF GridSearchCV#####################
rf_grid = GridSearchCV(
	estimator = RandomForestClassifier(warm_start=True, random_state=seed),
	param_grid={
		'n_estimators':[100,150,200],
		'criterion':['gini', 'entropy'],
		'max_features':[0.5, 0.7, 1.0, 10],
		'max_depth':[8, 9,10],
		'bootstrap': [True]
	},
	cv = kfold,
	scoring =scoring,
	n_jobs = processors)
rf_grid.fit(x,y)
print(rf_grid.best_score_)
print(rf_grid.best_params_)


#RF, BEST参数：{'n_estimators': 200,  'criterion': 'entropy', 'max_depth': 8 , 'max_feature':0.5}
'''

'''

###adaboost Gridsearch CV#################
ada_grid =GridSearchCV(
	estimator=AdaBoostClassifier(random_state=seed),
	param_grid={
	'algorithm':['SAMME', 'SAMME.R'],
	'n_estimators':[10,25,50],
	},
	cv= kfold,
	scoring = scoring,
	n_jobs =processors)
ada_grid.fit(x,y)
print(ada_grid.best_score_)
print(ada_grid.best_params_)

#adaboost，best参数：{'learning_rate': 0.001, algorithm': 'SAMME' , 'n_estimators': 10}
'''

'''
###GBDT GridSearchCV#################################
gbm_grid = GridSearchCV(
	estimator = GradientBoostingClassifier(warm_start=True, random_state=seed),
	param_grid = {
	'n_estimators': [100, 150, 200],
	'max_depth': [8, 9,10],
	'max_features': [0.5, 0.7,1.0],
	'learning_rate':[0.1, 0.5 ,1]

	},
	cv =kfold,
	scoring = scoring,
	n_jobs =processors)

gbm_grid.fit(x_train,y_train)
print(gbm_grid.best_score_)
print(gbm_grid.best_params_)

#GBDT, best 参数：{'learning_rate':0.1, 'max_depth':8, 'max_features': 0.5, 'n_estimators': 100}


####XGBT GridSearchCV#############
xgb_grid = GridSearchCV(
	estimator =XGBClassifier(),
	param_grid = {
	'n_estimators': [100, 150, 200],
	'max_depth': [6, 9],
	'learning_rate': [0.1 ,0.5, 1],
	},
	cv = kfold,
	scoring =scoring,
	n_jobs = processors)
xgb_grid.fit(x_train,y_train)
print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
# XGBT, best参数： {'learning_rate': 0.1, 'max_depth': 6, 'n_estimator': 100}
'''

########################Voting ensemble#########
#create sub models
estimators = []
estimators.append(('lr', LogisticRegression(penalty='l2', C=1000)))
estimators.append(('gbm', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=8, max_features=0.5)))
estimators.append(('rf', RandomForestClassifier(bootstrap=True, max_depth=8, max_features=0.5, criterion='entropy')))
estimators.append(('ada', AdaBoostClassifier(algorithm='SAMME', learning_rate=0.001, n_estimators=10, random_state=seed)))
estimators.append(('xgb', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)))

#create the ensemble model 
ensemble = VotingClassifier(estimators, voting='soft', weights=[2,3,3,1,3])
#CV for Voting ensemble
results = cross_val_score(ensemble, x_train, y_train, cv=kfold, scoring=scoring, n_jobs=processors)
print('ensemble model results:', '({0:.3f})+/-({1:.3f})'.format(results.mean(), results.std()))


##########################################################################
#######################生成结果，用于提交#################################
model= ensemble
model.fit(x_train, y_train)
pre_proba_test = model.predict_proba(x_test)
sample = pd.read_csv('F:\\kaggle_kobe\\sample_submission.csv')

for i in range(len(sample)):
	sample.iloc[i,1] = pre_proba_test[i][1]

sample.to_csv('F:\\kaggle_kobe\\test_submission.csv')
print('finished !')
#finished !