import warnings
warnings.filterwarnings("ignore")
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style = 'white',context = 'notebook',palette = 'muted')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('实验数据大小:',train.shape)
print('预测数据大小:',test.shape)

#将实验数据和预测数据合并
print('开始整合数据......')
full = train.append(test,ignore_index=True)

#计算不同类型embarked的乘客，其生存率为多少
# print('Embarked为"S"的乘客，其生存率为%.2f'%full['Survived'][full['Embarked']=='S'].value_counts(normalize=True)[1])
# print('Embarked为"C"的乘客，其生存率为%.2f'%full['Survived'][full['Embarked']=='C'].value_counts(normalize=True)[1])
# print('Embarked为"Q"的乘客，其生存率为%.2f'%full['Survived'][full['Embarked']=='Q'].value_counts(normalize=True)[1])
#结论：法国登船乘客生存率高

#法国登船乘客生存率较高原因可能与其头等舱乘客比例较高有关
#sns.factorplot('Pclass',col='Embarked',data=train,kind='count',size=3)

#计算Parch对生存率的影响
# sns.barplot(data=train,x='Parch',y='Survived')
#结论：当乘客同行的父母及子女数量适中时，生存率较高

#计算SibSp对生存率的影响
# sns.barplot(data=train,x='SibSp',y='Survived')
#结论：当乘客同行的同辈数量适中时生存率较高

#计算Pclass对生存率的影响
# sns.barplot(data=train,x='Pclass',y='Survived')
#结论：乘客客舱等级越高，生存率越高

#计算Sex对生存率的影响
# sns.barplot(data=train,x='Sex',y='Survived')
#结论：女性的生存率远高于男性

#探讨Age对生存率的影响
# ageFacet=sns.FacetGrid(train,hue='Survived',aspect=3)
# ageFacet.map(sns.kdeplot,'Age',shade=True)
# ageFacet.set(xlim=(0,train['Age'].max()))
# ageFacet.add_legend()
#结论8-10岁生存率较高

#探讨Fare对生存率的影响
# FareFacet=sns.FacetGrid(train,hue='Survived',aspect=3)
# FareFacet.map(sns.kdeplot,'Fare',shade=True)
# FareFacet.set(xlim=(0,train['Fare'].max()))
# FareFacet.add_legend()
#结论：当票价低于18左右时乘客生存率较低，票价越高生存率一般越高

#查看票价分布特征
# farePlot = sns.distplot(full['Fare'][full['Fare'].notnull()],label='skewness:%.2f'%(full['Fare'].skew()))
# farePlot.legend(loc='best')
# plt.show()
#结论：偏度：4.37较大，图中标明数据左偏，因此进行对数处理，防止不均匀
full['Fare'] = full['Fare'].map(lambda x:np.log(x) if x>0 else 0)

#数据预处理开始
print('开始进行数据清洗，填补缺失值......')
#数据清洗
#Cabin中缺失值用U填充
full['Cabin'] = full['Cabin'].fillna('U')
# print(full['Cabin'].head())
#Embarked用S补充(因为已有数据表明S登船的明显占据大部分)
# print(full['Embarked'].value_counts())
full['Embarked']=full['Embarked'].fillna('S')
#Fare缺失值用平均票价填充
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())

#特征工程
print('开始进行特征提取......')
#1.利用Name提取出旅客头衔Title，其头衔反应其身份，探讨身份对生存率影响
full['Title'] = full['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())
# print(full['Title'].value_counts())
#将类似的title整合
TitleDict={}
TitleDict['Mr']='Mr'
TitleDict['Mlle']='Miss'
TitleDict['Miss']='Miss'
TitleDict['Master']='Master'
TitleDict['Jonkheer']='Master'
TitleDict['Mme']='Mrs'
TitleDict['Ms']='Mrs'
TitleDict['Mrs']='Mrs'
TitleDict['Don']='Royalty'
TitleDict['Sir']='Royalty'
TitleDict['the Countess']='Royalty'
TitleDict['Dona']='Royalty'
TitleDict['Lady']='Royalty'
TitleDict['Capt']='Officer'
TitleDict['Col']='Officer'
TitleDict['Major']='Officer'
TitleDict['Dr']='Officer'
TitleDict['Rev']='Officer'

full['Title']=full['Title'].map(TitleDict)
# print(full['Title'].value_counts())
#探讨Ttlte对生存率的影响
# sns.barplot(data=full,x='Title',y='Survived')
# plt.show()
#结论：头衔为'Mr'及'Officer'的乘客，生存率明显较低

#2.整个家庭人数FamilyNum特征
full['familyNum'] = full['Parch']+full['SibSp']+1
#探讨familyNum对生存率的影响
# sns.barplot(data=full,x='familyNum',y='Survived')
# plt.show()
#结论：家庭成员人数在2-4人时，乘客的生存率较高

#因此将2-4划分为中等家庭，4+为大，2-为小
def familysize(familyNum):
    if familyNum==1:
        return 0
    elif (familyNum>=2)&(familyNum<=4):
        return 1
    else:
        return 2
full['familySize']=full['familyNum'].map(familysize)
# print(full['familySize'].value_counts())
#探讨familySize对生存率的影响
# sns.barplot(data=full,x='familySize',y='Survived')
# plt.show()
#结论：当家庭规模适中时，乘客的生存率更高

#3.客舱类型，Cabin首字母反映了客舱类型
full['Deck']=full['Cabin'].map(lambda x:x[0])
#探讨Deck对生存率的影响
# sns.barplot(data=full,x='Deck',y='Survived')
# plt.show()
#结论：当乘客的客舱类型为B/D/E时，生存率较高；当客舱类型为U/T时，生存率较低

#4.同一票号的乘客数量TickCot和TickGroup
TickCountDict=full['Ticket'].value_counts()
# print(TickCountDict.head())
full['TickCot']=full['Ticket'].map(TickCountDict)
#探究TickCot对Survived的影响
# sns.barplot(data=full,x='TickCot',y='Survived')
# plt.show()
#结论：当TickCot大小适中时，乘客生存率较高
#与家庭大小一致，将其分为三类，高中低
def TickCountGroup(num):
    if (num>=2)&(num<=4):
        return 0
    elif (num==1)|((num>=5)&(num<=8)):
        return 1
    else :
        return 2
full['TickGroup']=full['TickCot'].map(TickCountGroup)
#探究TickGroup对Survived的影响
# sns.barplot(data=full,x='TickGroup',y='Survived')
# plt.show()

#Age缺失值填充，构建随机森林模型预测缺失的数据
#查看Age与Parch、Pclass、Sex、SibSp、Title、familyNum、familySize、Deck、
# TickCot、TickGroup等变量的相关系数大小，筛选出相关性较高的变量构建预测模型

print('开始利用随机森林模型填补Age缺失数据......')
#1.筛选数据
AgePre = full[['Age','Parch','Pclass','SibSp','Title','familyNum','TickCot']]
#进行one-hot编码
AgePre=pd.get_dummies(AgePre)
ParAge=pd.get_dummies(AgePre['Parch'],prefix='Parch')
SibAge=pd.get_dummies(AgePre['SibSp'],prefix='SibSp')
PclAge=pd.get_dummies(AgePre['Pclass'],prefix='Pclass')
#查看变量间的相关性
AgeCorrDf = AgePre.corr()
AgeCorrDf['Age'].sort_values()
# print(AgeCorrDf['Age'])
#拼接数据
AgePre=pd.concat([AgePre,ParAge,SibAge,PclAge],axis=1)
# print(AgePre.head())
#拆分数据并建立模型
AgeKnown = AgePre[AgePre['Age'].notnull()]
AgeUnKonwn = AgePre[AgePre['Age'].isnull()]
#生成实验数据的特征和标签
AgeKnown_X=AgeKnown.drop(['Age'],axis=1)
AgeKnown_Y=AgeKnown['Age']
#生成预测数据的特征
AgeUnKnown_X=AgeUnKonwn.drop(['Age'],axis=1)

#利用随机森林构建模型
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=None,n_estimators=500,n_jobs=-1)
rfr.fit(AgeKnown_X,AgeKnown_Y)

#模型得分
# print(rfr.score(AgeKnown_X,AgeKnown_Y))
#预测年龄
AgeUnKnown_Y=rfr.predict(AgeUnKnown_X)
#填充预测数据
full.loc[full['Age'].isnull(),['Age']]=AgeUnKnown_Y
# full.info()  #此时已无缺失值

# full.to_csv('test.csv')
#同组识别
print('开始进行同组识别......')
#提取两部分数据，分别查看其“姓氏”是否存在同组效应
# （因为性别和年龄与乘客生存率关系最为密切，因此用这两个特征作为分类条件）
full['Surname'] = full['Name'].map(lambda x:x.split(',')[0].strip())
SurNameDict = full['Surname'].value_counts()
full['SurnameNum'] = full['Surname'].map(SurNameDict)
#数据分为两组
#1.12岁以上男性：找出男性中同姓氏均获救的部分
MaleDf = full[(full['Sex']=='male')&(full['Age']>12)&(full['familyNum']>=2)]
#2.女性以及年龄在12岁以下儿童
FemChildDf = full[((full['Sex']=='female')|(full['Age']<=12))&(full['familyNum']>=2)]
#男性同组效应分析
MSurNamDf = MaleDf['Survived'].groupby(MaleDf['Surname']).mean()
# print(MSurNamDf.head())
# print(MSurNamDf.value_counts())
#获取生存率为1的姓氏
MSurNamDict=MSurNamDf[MSurNamDf.values==1].index

#分析女性及儿童同组效应
FCSurNamDf=FemChildDf['Survived'].groupby(FemChildDf['Surname']).mean()
# print(FCSurNamDf.value_counts())
#获取生存率为0的姓氏
FCSurNamDict=FCSurNamDf[FCSurNamDf.values==0].index

#进行修正
print('开始根据同组识别结果对数据进行修正......')
#男性数据修正为：1、性别改为女；2、年龄改为5；
#女性及儿童数据修正为：1、性别改为男；2、年龄改为60
full.loc[(full['Survived'].isnull())&(full['Surname'].isin(MSurNamDict))&(full['Sex']=='male'),'Age'] =5
full.loc[(full['Survived'].isnull())&(full['Surname'].isin(MSurNamDict))&(full['Sex']=='male'),'Sex']='female'

full.loc[(full['Survived'].isnull())&(full['Surname'].isin(FCSurNamDict))&((full['Sex']=='female')|(full['Age']<=12)),'Age']=60
full.loc[(full['Survived'].isnull())&(full['Surname'].isin(FCSurNamDict))&((full['Sex']=='female')|(full['Age']<=12)),'Sex']='male'

#筛选子集
print('开始筛选子集，剔除相关性较低特征......')
#通过找出与乘客生存率“Survived”相关性更高的特征，剔除重复的且相关性较低的特征
#人工筛选
fullSel = full.drop(['Cabin','Name','Ticket','PassengerId','Surname','SurnameNum'],axis=1)
#查看相关性
corrDf = fullSel.corr()
corrDf['Survived'].sort_values(ascending=True)
# print(corrDf)
#热力图图示化相关性
# plt.figure(figsize=(8,8))
# sns.heatmap(corrDf,cmap='BrBG',annot=True,linewidths=0.5)
# plt.xticks(rotation=45)
# plt.show()
#人工初步筛除与标签预测明显不相关或相关度很低的特征
fullSel = fullSel.drop(['familyNum','SibSp','TickCot','Parch'],axis=1)

fullSel=pd.get_dummies(fullSel)
PclassDf=pd.get_dummies(fullSel['Pclass'],prefix='Pclass')
TickGroupDf=pd.get_dummies(full['TickGroup'],prefix='TickGroup')
familySizeDf=pd.get_dummies(full['familySize'],prefix='familySize')

fullSel=pd.concat([fullSel,PclassDf,TickGroupDf,familySizeDf],axis=1)
#模型选择
print('开始建立模型......')
#拆分实验数据与预测数据
experData=fullSel[fullSel['Survived'].notnull()]
preData=fullSel[fullSel['Survived'].isnull()]

experData_X=experData.drop('Survived',axis=1)
experData_Y=experData['Survived']
preData_X=preData.drop('Survived',axis=1)

#导入机器学习算法库
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold

#设置kfold，交叉采样法拆分数据集
kfold=StratifiedKFold(n_splits=10)

#汇总不同模型算法
classifiers=[]
classifiers.append(SVC())
classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier())
classifiers.append(ExtraTreesClassifier())
classifiers.append(GradientBoostingClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())
classifiers.append(LinearDiscriminantAnalysis())

# 不同机器学习交叉验证结果汇总
print('开始使用各种模型进行机器学习......')
cv_results=[]
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier,experData_X,experData_Y,scoring='accuracy',cv=kfold,n_jobs=-1))

#求出模型得分的均值和标准差
print('开始计算各模型均值和标准差......')
cv_means=[]
cv_std=[]
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

#汇总数据
cvResDf=pd.DataFrame({'cv_mean':cv_means,
                     'cv_std':cv_std,
                     'algorithm':['SVC','DecisionTreeCla','RandomForestCla','ExtraTreesCla',
                                  'GradientBoostingCla','KNN','LR','LinearDiscrimiAna']})
print('各模型结果为：')
print(cvResDf)
print('结果发现GBC与LR模型能够进行进一步优化......')
#可视化结果
# cvResFacet=sns.FacetGrid(cvResDf.sort_values(by='cv_mean',ascending=False),sharex=False,sharey=False,aspect=2)
# cvResFacet.map(sns.barplot,'cv_mean','algorithm',**{'xerr':cv_std},palette='muted')
# cvResFacet.set(xlim=(0.7,0.9))
# cvResFacet.add_legend()

#GradientBoostingClassifier模型
print('开始指定建立GBC模型......')
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["log_loss"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1]
              }
modelgsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold,scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsGBC.fit(experData_X,experData_Y)

#LogisticRegression模型
print('开始指定建立LR模型......')
modelLR=LogisticRegression()
LR_param_grid = {'C' : [1,2,3],'penalty':['l1','l2']}
modelgsLR = GridSearchCV(modelLR,param_grid = LR_param_grid, cv=kfold,scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsLR.fit(experData_X,experData_Y)

#modelgsGBC模型
print('modelgsGBC模型得分为：%.3f'%modelgsGBC.best_score_)
#modelgsLR模型
print('modelgsLR模型得分为：%.3f'%modelgsLR.best_score_)

#查看模型ROC曲线
#求出测试数据模型的预测值
modelgsGBCtestpre_y=modelgsGBC.predict(experData_X).astype(int)
#画图
# from sklearn.metrics import roc_curve, auc  ###计算roc和auc
# fpr,tpr,threshold = roc_curve(experData_Y, modelgsGBCtestpre_y) ###计算真正率和假正率
# roc_auc = auc(fpr,tpr) ###计算auc的值
#
# plt.figure()
# lw = 2
# plt.figure(figsize=(10,10))
# plt.plot(fpr, tpr, color='r',lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Titanic GradientBoostingClassifier Model')
# plt.legend(loc="lower right")
# plt.show()

#TitanicGBSmodle
print('利用GBC进行预测......')
GBCpreData_Y=modelgsGBC.predict(preData_X)
GBCpreData_Y=GBCpreData_Y.astype(int)
#导出预测结果
print('将结果导出为csv文件......')
GBCpreResultDf=pd.DataFrame()
GBCpreResultDf['PassengerId']=full['PassengerId'][full['Survived'].isnull()]
GBCpreResultDf['Survived']=GBCpreData_Y
#将预测结果导出为csv文件
GBCpreResultDf.to_csv('TitanicGBSmodle.csv',index=False)
print('任务结束......')