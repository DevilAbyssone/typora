## Matplotlib
```pyhon
import Matplotlib.pyplot as plt
import random
# 生成数据
x= range(60)
y= [random.uniform(10,15)for i in x]
# 创建画布
plt.figure(figsize(20,8),dpi = 100)#画布大小
# 图像绘制
plt.plot(x,y)# color
# 绘制x、y轴刻度
y_ticks = range(40)
plt.yticks(y_ticks[::5]) # 以5分割 0 5 ....30
x_ticks_label = ['11点{}分'.format(i) for i in x]
plt.xticks(x[::5],x_ticks_label[::5])# 用label来替换x的数字
# 11点0分 ...11.55分
plt.grid(True, linestyle="--", alpha=1)# 添加网格 linestyle款式alpha深浅
#标题
plt.xlabel() plt.ylabel() plt.title("",fonntsize=20)
# 多次plot可以实现线的叠加
#图例
plt.legend(loc="best")# 位置
plot.show() # 先用plt.savafig()再用plt.show()
```
### 用subplots画图
```python
form Matplotlib.pyplot import subplots
import random
# 面向对象画图
#数据部分
x= range(60)
y_shanghai=[random.uniform(15,18)for i in x]
y_beijing=[random.uniform(1,14)for i in x]
#画布
fig.axes = plt.subplots(nrows=1, ncols=2, figsize=(20,8), dpi=100)
axes[0].plot(x,y_shanghai,label='')
axes[1].plot(x,y_beijing,label='')
axes[0].set_xticks()
axes[0].set_yticks()
axes[0].set_xticks(x[::5],x_ticks_label[::5])# 替换规则同样适用
axes[0].grid(True, linestyle='',alpha=0.5)
axes[0].set_xlabel/ylabel/title
axes[0].lengend()
plot.show()
```
### 其他函数
```python
plt.plot()#折线图 
plt.scatter()#散点图
plt.bar()#柱状图  
plt.hist()直方图#
plt.pie()#饼状图
# sinx曲线
import numpy as np
x = np.linspace(-10,10,1000) #取-10到10中1000个数
y= np.sin(X)
plt.figure(figsize=(20,8),dpi=100)
plt.plot(x,y)
plt.show()
```
## Numpy
### np
```python
import time
%time ...#代码前面加time获取时间

import numpy as np
a = np.array([1, 2, 3])
#ndarray（可以用np去点这些方法） .shape维度 .ndim维度 .dtype类型 .size元素数量 .itemsize元素的长度
#全0数组
np.zero([行, 列])
#全1数组
np.ones([行, 列])
#10-50每间隔两个数选一个数
np.arraay(10,50,2)
# 0-1均匀分布
np.random.rand(2,3)
np.random.uniform(low =1, high=10, size=(3,5))
#生成幂级数的数 生成以10的N次幂的数据
np.longspace(0,2,3)
#正态分布 loc均值 scale标准差 size生成多少数据
np.random.normal(loc = ,scale= ,size = None)
stock_cahnge = np.random.normal(0,1,(8,10))
stock_change[0:1,0:3]# 行，列
#三维定位
a1 = np.array[[[1,2,3],[4,5,6] [[12,3,34],[5,6,7]]]
# a1[0,0,1] 2
stock_change.reshape([10,8])# reshape方法先将所有的数据排成行在进行分 所以就算互换行列位置不会转置
			.reshape([-1,10])# 一行10个，不确定行
stock_change.astype(type)# int 32
			.tostring()#变成bytes
np.unique()#去重
stock_change[stock_change >1]=2 #将大于1的所有值都赋予2
np.all(data[0:2,0:5]>0)  # 判断所有  返回False/True 
np.any(data[0:5],:>0)# 有一个满足条件都是True 类似于ε
np.where(temp>0,1,0)# 找出temp中大于0赋值给1
np.where(np.logical_and/logical_or(temp<5,temp>1),1,0)# 双条件赋值
np.min()/max/median/mean/std/var
argmax argmin #返回最大 最小的下标
```
### ndarray运算
广播机制 维度相同或者shape对应位置为1才能操作
矩阵相乘
```python
np.matul(a,b) np.dot(a,b)#两者方法上得出的结果是一致的但是np.matul不支持a或者b为一个具体数字而dot方法支持
```
## pandas
### 基础pandas
``` python
import pandas as pd
stock_day_rise = pd.Dataframe(stock_change)
stock_code=['股票'+str(i) for i in range(stock_day_rise.shape[0])] #stock_day_rise.shape[0]表示长度 会返回一个列的数量
data=pd.data_range('2017-01-01',periods = stock_day_rise.shape[0])
#start:开始时间 end:结束时间 period:时间天数 freq='B' b默认跳过周末
pd.Dataframe(stock_change,index = stock_code,colums = data)# index列索引 colums行索引
#pd内置方法
index() colums() T value head() tail()
set_index(keys,drop = True)# 以某列设置新的索引 原来的索引丢掉 不丢会出现两个列索引 设置两个索引 multiIndex
# data[""][""]先列后行索引 一定要具体列名 data[:1,:2]不行
.loc #取列 下标值和具体值都可以
.ix  #混合下标和列名拿值
.iloc
data.sort_values(by='',ascending =False)
data.sort_values(by=["",""])
	.sort_index()
```
### 算数运算
```pyhton
data[""].add(10)
data[""]+100
```
### 逻辑运算
```pyhton
< > | ?
# 取open列中大于23的值
data["open">23]
data[(data["open">23) & (data["open"]<24)]
#一定注意加括号 符号优先级问题！
data.query()
data[data["turnover"isin([4.19,2.39])]]
```
### 统计运算
```pyhton
discribe  mode()众数  idmax  idmin 返回下标索引值
cumsum cummax cummin cumprod(乘积) 累计n个数的
自定义函数
apply（func，axis = 0）
data[["open","close"]].apple(lambda x:x.max()-x.min(),axis = 0)
#pandas提供画图函数
DataFrame.plot(x,y) #bar barh(横着的hist) hist pie scatter
```
### 读取数据
```python
pandas.read_csv(filepath,sep='')# usecols:[""] index=Fales/True
pandas.to_csv(filepath.sep='', colums= , header=, index= )
#mode = 'w' encoding = 默认分割. w表示重写 a表示追加
#hdf5 压缩方式存储 读取效率快 节省空间 支持跨平台
pandas.read_hdf(path,key=)#key为自己设置的密码
pandas.to_hdf(path,key=)
pandas.read_json(path,orient= ,typ= , lines=Ture)#orient方式typ框架
```
### 处理缺失值
```python
# NAN是属于float类型
pd.isnull(df)
  .notnull(df)
np.any(pd.isnull())
np.all(pd.notnull())
 dorpna(axis='rows')#丢掉有缺失值的行
 fullna(value, inplace=True)#填补替换NA值
 # 用平均值替换
value=data.mean()
# 当出现标记值？的时候 用nan替换？
data = data.replace(to_replace = '?', value= np.NaN)
```
### 离散化处理
```python
pd.qcut(data,bins)#bins分的组数也可以使用自定义区间如 bins=[-100,-7,-5,0,3,5,7,100]
qcut.value_counts()#统计每组的分的个数
# one-hot编码
pandas.get_dummies(data,prefix = None)#prefix分组名称
dummies = pd.get_dummies(p_counts,prefix='rise')
```
### 合并
```python
pd.concat([data1,data2], axis= 1)#行索引进行合并 注意要索引一样
pd.merge(left,right,on=['key1','key2'])# how = inner/outer 内连接和外连接
```
### 分组聚合
```python
data.groupby([""])
# cpu:IO密集型 gpu:计算密集型
#大列表中取小列表 可以用for的嵌套和列表表达式
for i in list:
	for j in i:

[i for j in list for i in j]
```
## 标准化变量
$$
\begin{align}
x^*=\frac{x-m}{s}\quad s为标准差\quad m为平均值
\end{align}
$$
### 距离公式
	+ 欧式距离
	+ 曼哈顿距离
	+ 切比雪夫距离
	+ 闵式距离
	+ 标准化的欧氏距离
	+ 余弦距离
	+ 汉明距离
	+ 杰卡德距离
	+ 马氏距离
## K近邻
k过大 过拟合 受到异常点影响
k过小 欠拟合 受样本均衡问题
kd树解决快速进行k近邻搜索
### api
```python
sklearn.neighbors.kNeighboraClassifier(n_neighbors= 5, algorithm='')
# algorithm: 'auto''ball_tree'(维度大于20时使用，超平面类似于球体) 'kd_tree' 'brute'
优点：简单有效 重新训练代价较低 适合类域交叉样本 适合大样本自动分类
缺点：惰性学习 类别评分不是规格化 输出可解释性不强 对不均衡样本不擅长
```
### K-means api
```python
sklearn.cluster.KMeans(n_clusters=8)
estimator.fit(x) .pridict(x) .fit_pridict
#评估
from sklearn.metrics import calinski-Harabarabaz_score
calinski_harabaz_score(x,y_pred)# CH评估
```
K-means k值
+ SSE误差平方和
+ 手肘法
+ 轮廓系数法
+ CH系数 类别内部系数的距离平方差越小越好  类别间的距离平方和越大越好
优化算法 Canopy 抗干扰 k值精准 减少计算量 但会陷入局部最优解
kmeans++ 二分K-means k-medoids kernel K-means
## 数据操作
### 加载数据集问题
```python
form sklearn.datasets import load_iris,fetch_20newsgroups 
iris = load_iris() #小数据集
news= fetch_20newsgroups()#大数据集
#seaborn库比matplotlib好看
import seaborn
seaborn.implot()
# 参数：x,y data hue=‘target’(目标值是什么) fit_reg()是否有辅助拟合曲线
```
### 归一化
$$
\begin{align}
& x'= \frac{x-min}{max-min}\\
& x"=x'(mx-mi)+mi\quad mx为1\quad mi为0
\end{align}
$$
#### api
```python
from aklearn.processing import MinMaxScaler
transfer =MinMaxScale(feature_range(0,1))#实例化转化器
data1 = transfer.fit_transform(data[['','']])
# 归一化容易受到异常点的影响，稳定性(鲁棒性)较差 用来统计传统精准小数据
```
### 标准化
$$
\begin{align}
 x'=\frac{x-mean}{\sigma}
\end{align}
$$
#### api
```pyhton
#异常值影响值较小 适合大样本数据
from sklean.preprocessing import StandardScalar
transfer = StandardScalar()
data1 = transfer.fit_transform(data[['','']])
```
### 数据分割
```python
from sklearn.model_selection import train_test_split
x_train,x_text,y_train,y_test = train_test_split(data, target, random_state=22, test_size=0.2 )
```
### 训练模型
```python
estimator = KNeighborsClassfier(n_neighbors = 5)
estimator.fit(x_train,y_train)
```
### 输出预测值
```python
y_pre = estimator.predict(x_test)
```
### 精度
```pyhton
ret = estimator.score(x_test,y_test)
```
### (折)交叉验证、网格搜索
可以让模型更加准确可信，但是准确度不能提高
```python
from sklearn.model_selection import GridSearchCV
GridSearchCV(estimator, param_grid=None, CV=None)
# estimator估计器对象 param_grid超参数 {"n_neighbors":{1,3,5}} CV 指定几折交叉运算[折的是训练集] n_jobs = -1 cpu满负载运行
estimator.best_estimator
		 .best_score_
		 .cv_results_
```
### 模型的加载和保存
```python
from sklearn.externals import joblib
joblib.dump(estimator,'test.pkl')
estimator=joblib.load('text.pkl')
```
## 线性回归
```python
from sklearn.linear_model import LinearRegeression
estimator = LinearRegression()# fit_intercep= True 计算偏置
estimator.fit(x,y)
coef = estimator.coef_
estimator.intercept# 表示偏置
estimator.pridict([[80,100]])
```
损失函数
$$
\begin{align}
& J(\theta)=\sum\limits_{i=1}^{m}({h_w(x_i)-y_i})^2
\end{align}
$$
损失与优化
+ 用正规方程来找到最优$\omega$减小$J(\theta)$​   此方法在样本较小时使用
  $$
  \begin{align}
  损失函数：
  \\ & J(\theta)=(y-x_\omega)^2\\
  正规方程：
  \\& \omega =(x^Tx)^{-1}x^Ty
  \end{align}
  $$
  证明过程如下所示：
  ​                                                   对$\omega$​​求导：
$$
\begin{align}
2(y-x_\omega)*(-x)&=0
\\2(x_\omega-y)*(xx^T)&=0X^T
\\2(x_\omega-y)*(xx^T)(xx^T)^{-1}&=0X^T(xx^T)^{-1}\quad注意不可知x是否可逆不可直接求逆，
\\x\omega&=y\qquad\qquad\qquad因此需要利用x^Tx是方阵
\\x^Tx\omega&=x^Ty
\\(xx^T)^{-1}(xx^T)*\omega&=(xx^T)^{-1}*x^Ty
\\\omega& =(x^Tx)^{-1}x^Ty\qquad得到正规方程
\end{align}
$$
+ 梯度下降
  exp:
	​			
	$$
	\begin{align}
	假设目标函&数如下所示：
	\\j(\theta)&=\theta^2
	\\j'(\theta)&=2\theta
	\\初始点：\theta^0=1\quad&学习率:\alpha=0.4
	\\\theta^0=1 \quad \theta^1=\theta^0-\alpha*J'&(\theta^0)=1-0.4*2*1=0.2 
	\\同理可得\quad\theta&^3=0.008\quad\theta^4=0.0016
	\\\theta^i=\theta^i-&\alpha\frac{\partial}{\partial\theta^i}j(\theta)
	
	\end{align}
	$$
  但是这样并不能保证找到最优解，只保证满足所给条件。
+ 总结
	+ 小规模数据 LinearRegression(不能解决拟合)和岭回归
	+ 大规模数据 SGDRegressor梯度下降
### 梯度下降算法
+ 全梯度下降算法(FG)计算所有样本梯度，速度较慢
+ 随机梯度下降算法(SG)取一个样本进行迭代，遇上噪声容易陷入局部最优解
+ 随机平均梯度下降算法(SAG)计算样本权重，每一次维持这个权重不变，初期较慢
+ 小批量梯度下降算法(mini-bantch)将一部分样本进行梯度下降
#### 梯度下降算法优化算法
+ SAG with momentum
用SAG算法求对过去的K次的梯度平均值，是对过去所有梯度的加权平均
+ Adagrad
让学习率成为可以改变的使用参数，对于出现次数较少的特征时增大学习率
+ Adadelta
Adagard的扩展算法，以处理学习率单调递减问题
+ RMSProp
结合了梯度平方的指数，移动平均数来进行调整学习率。在不稳定(Non-stationary)的目标函数可以很好的收敛
Adam
结合AdaGrad、RMSProp算法 自适应学习率算法
### 均方误差(MSE)
$$
MSE=\frac{1}{m}\sum\limits_{i=1}^{m}({y_i-y})^2
$$
```python
from sklean.metrics import mean_squard_error
ret = mean_squard_error(y_test,y_pre)
# 使用梯度下降方法
sklearn linear_model.SGDRegressor(loss="quard_loss", fit_intercept=True, learning_rate ='invscalling', eta0=0.01)
# 方法.coef_回归系数       学习率参数 "constant": eta =eta0 不变  "optional": eta = 1.0/(alpha*(t+t0))
#	  .intercept_偏置                "invscalling": eta = eta0/pow(t,power_t)   power_t：0.25
```
+ 欠拟合
	+ 增加特征项
	+ 添加多项式特征
+ 过拟合
	+ 重新清洗数据	
	+ 增加数据的训练量	
	+ 正则化	
	+ 减少特征维度，防止维灾难（随着维度增加，分类器性能逐步上升，但是达到某点之后，其性能就会逐渐下降）
	
	  |        | training data | testing data |
	  | :----: | :-----------: | :----------: |
	  | 欠拟合 |       √       |      √       |
	  | 过拟合 |       ×       |      ×       |
### 正则化
+ L1正则化
使一些$\omega$权值直接为0，删除特征的影响 LASSO回归
+ L2正则化
+使一些$\omega$都很小，接近于0，消除某个特征的影响
越小参数模型越简单，不容易过拟合          Ridge回归
#### Ridge Regression 岭回归
线性回归中的cost function 中增加了L2正则项(regularization term)
岭回归中的代价函数：
$$
\begin{align}
J(\theta)=MSE(\theta)&+\alpha\sum\limits_{i=1}^{m}{\theta_i^2}
\\即J(\theta)=\frac{1}{m}\sum\limits_{i=1}^{m}&(\theta^T*x^{(i)-y^{(i)}})^2+\alpha\sum\limits_{i=1}^{m}{\theta_i^2}
\end{align}
$$
当$\alpha$为0时，则为线性回归

#### LASSO回归
$$
sign(\theta_i)=
\begin{cases}
-1,&\text{if }\theta_i\text{ <0}\\
0,&\text{if }\theta_i\text{ =0}\\
1,&\text{if }\theta_i\text{ >0}\\
\end{cases}
$$
LASSO回归中的代价函数：
$$
\begin{align}
J(\theta)=MSE(\theta)&+\alpha\sum\limits_{i=1}^{m}{|\theta_i|}
\\即J(\theta)=\frac{1}{m}\sum\limits_{i=1}^{m}&(\theta^T*x^{(i)-y^{(i)}})^2+\alpha\sum\limits_{i=1}^{m}{|\theta_i|}
\end{align}
$$
值得注意的是$|\theta_i|$在顶点的时候是不可导的，所以在$\theta_i$=0中用次梯度向量代替梯度
$$
g(\theta,J)=\nabla_\theta MSE(\theta)+\left(\begin{matrix}
sign(\theta_i)\\
...\\
sign(\theta_n)
\end{matrix}\right)
$$
$\alpha$​较大时可以让高阶多项式退化为二阶甚至线性函数 权重直接被清0
#### 弹性网络(Elastic Net)
$$
\begin{align}
J(\theta)&=MSE(\theta)+r\alpha\sum\limits_{i=1}^{m}{|\theta_i|}+\frac{1-r}{2}\alpha\sum\limits_{i=1}^{m}{\theta_i^2}\\
r&=
\begin{cases}
0,&\text{Ridge回归 }\\
1,&\text{Lasso回归 }\\
\end{cases}
\end{align}
$$
常用Ridge回归 Lasso回归不稳定
少部分特征用弹性网络 特征维度高于训练样本或特征值强相关
#### api
from sklean.linear_model import Ridge,ElasticNet,Lasso
sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, solver= "auto", ,normalize= False)alpha正则化力度 0-1 1-10 solver优化方式 SAG auto
nomalize是否进行标准化
正则化力度越大 权重越低
### Early Stopping(早停法)
当我们训练深度学习神经网络的时候通常希望能获得最好的泛化性能（generalization performance，即可以很好地拟合数据）。但是所有的标准深度学习神经网络结构如全连接多层感知机都很容易过拟合：当网络在训练集上表现越来越好，错误率越来越低的时候，实际上在某一刻，它在测试集的表现已经开始变差。
https://blog.csdn.net/df19900725/article/details/82973049
## 逻辑回归 
解决二分类问题

### 输入 
$h(\omega)=\omega_1x_1+\omega_2x_2...+b $​​​​
### 激活函数
sigmoid函数
$$
g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}
$$
![](https://raw.githubusercontent.com/DevilAbyssone/typora/main/img/20210723195457.png)

### 损失函数
$$
\begin{align}
cost(h_\theta(x),y)&=
\begin{cases}
-log(h_\theta(x)),&\text{if }y\text{ =1}\\
-log(1-h_\theta(x)),&\text{if }y\text{ =0}\\
\end{cases}
\\cost(h_\theta(x),y)&=\sum\limits_{i=1}^{m}(-y_ilog(h_\theta(x)))-(1-y_i)log(1-h_\theta(x))
\end{align}
$$
损失函数优化
提升原本属于1的概率 降低原本属于0的概率

### api
```pyhton
sklearn.linear_model.LogisticRegression(solver='liblinear', penalty='12', c=1.0)
# solver: liblinear 小数据集 sag saga 大数据集 sag saga newton-cg lbfgs多分类可以处理多项损失 liblinear仅限2分类
# penalty正则化种类 c正则化力度 回归与分类api可以混合使用
#LogisticRegression相当于SGDClassifier(loss = "log", penalty="L2")
```
### 混淆矩阵
|      | 正例 | 负例 |
| :--: | :--: | :--: |
| 正例 |  TP  |  FN  |
| 负例 |  FP  |  TN  |
准确率:(TP+TN)/(TP+TN+FN+FP)
精确率:TP/(TP+FN)
召回率:TP/(TP+FN)
F1-Score=2TP/(2TP+FN+FP)
```python
sklean.metrics.classification_report(y_true, y_pred, label=[], target_name=None)
```
labels指定类别对应数字不一定是0/1 target_names目标类别名称，现实里或者表格中想要叫的名字
TPR = TP/(TP+FN)所有预测1中1的比例
FPR = FP/(FP+TN)所有预测0中1的比例
ROC曲线：横坐标就是FPR，而纵坐标就是TPR，因此可以想见，当 TPR越大，而FPR越小时，说明分类结果是较好的。因此充分说明ROC用于二分类器描述的优势。
AUC 即ROC曲线下的面积，计算方式即为ROC Curve的微积分值，其物理意义可以表示为：随机给定一正一负两个样本，将正样本排在负样本之前的概率，因此AUC越大，说明正样本越有可能被排在负样本之前，即分类额结果越好。
+ ROC 可以反映二分类器的总体分类性能，但是无法直接从图中识别出分类最好的阈值，事实上最好的阈值也是视具体的场景所定；
+ ROC Curve 对应的AUC越大（或者说对于连续凸函数的ROC曲线越接近(0,1) )说明分类性能越好;
+ ROC曲线一定是需要在 y = x之上的，否则就是一个不理想的分类器
```python

form sklearn.metrics import roc_auc_score
sklean.metrics.coc_auc_score(y_ture/y_test, y_score/y_pre)
## 特征提取
sklearn.feature_extraction.text.CountVectorizer(stop_word[])
transfer = CountVectorizer()
data = transfer.fit_trabsform(data)
data.tostring()# 抽取结果
trnsfer.get_feature_names()# 特征名
### TF-IDF
sklearn.feature_extraction.text import TfidfVector
```
## 决策树
### api
from sklearn.tree.DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=None)
criterion gini(CART算法) entropy信息增益
min_sample_split# 最小子树划分样本数
min_sample_leaf# 叶子结点最小样本数
max_depth# 决策树最大深度
## 集成学习
boosting（弱的加起来变强，解决欠拟合）
bagging（强的集成，解决过拟合）
boosting 根据前一轮学习结果调整数据的重要性 对学习器进行加权投票 串行学习有先后
bagging 对数据进行采样训练 对所有学习器平均投票 并行学习，没有依赖
### api
```python
from sklearn.ensemble import AdaboostClassifier
```
GBDT = 梯度下降 +Boosting +决策树
XGBoosting = BOOSting +二阶泰勒展开 +Boosting
GBDT:
+ 使用梯度下降算法优化代价函数
+ 使用一层决策树作为弱学习器 负梯度作为目标值
+ 利用boosting思想进行集成
XGBoost:
+ boosting来对多个弱学习器进行迭代式学习
+ 二阶泰勒展开对损失函数使用 使用一阶和二阶梯度进行优化
+ 防止过拟合，加上惩罚项限制决策树叶子结点个数以及其值大小

![](https://img-blog.csdnimg.cn/20200409183923672.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTc1MzkyNA==,size_16,color_FFFFFF,t_70)







