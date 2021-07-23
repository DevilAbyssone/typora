## 字符串翻转
```pyhton
str1 = 'hello word'
print(str1[::-1])

form functools import reduce
print(reduce(lambda x,y,y+x,str1))
```
## 判断字符串是否回文
```python
def fun(string):
	if string == string(::-1)
		print("回文")
	else:
		print("不是回文")
```
## 单词大小写
```pyhton
str.title() #所有单词首字母大写
str.upper()#所有字母大写
str.capitalize()#字符串首字母大写
```
## 字符串拆分
```python
str.split()#返回列表
str.split("/")
str.strip()#去除左右两边的空格
```
## 合并字符串
```python
" ".join(list)
import re #去除字符串中不需要的字符
" "join(re.split("\W+",str)
```
## 寻找字符串中的唯一的元素
```python
''.join(set(str))

#对于列表的筛查
list(set(list))
```
## 将元素进行重复
```python
str/list*2

str1 = "python"
list1 = [1,2,3]
str2=""
list2=[]
for i in range(2):
	str2 += str1
	list1.extend(list1)
```
## 基于列表的扩展
```python
list1 = [2,2,2,2]
[2*x for x in list1]
#[4,4,4,4]
#列表展开
list1 = [[1,2,,3],[1,2]]
[i for k in list1 for i in k]
```
# 将列表展开
```python
form iteration_utilities import deepflatten
list1 = [1,2[2,1,[3,4]],[4,5],6,6]
deepflatten(list1)

def flatten(lst):
	res = []
	for i in lst:
		if  isinstance(i,list):
			res.extend(flatten(i))
		else:
			res.append(i)
		return res
```
## 二值交换
```python
a,b = b,a
```
## 统计列表中元素的频率
```pthon
form collections impore Counter
list1 = [1,1,2,2,3,3,3,3,3]
count = Counter(list1)
count["1"]
count.most_common(1)

dict1 = {}
for i in list1:
	if i in dict1:
		dict1[i] += 1
	else:
		dict[i] = 1
print(max(dict1,key = lambda x:dict1[x]))
```
## 判断字符串中的元素是否相同
```python
a = Counter(str1)
b = Counter(str2)
a == b?1,0
```
## 将数学字符串转化转化成列表
```python
str1 = '123'
list1 = list(map(int ,str1))

list1 - [int(i) for i in str1]
```
## try except finally
```python
try:
except AttributeError as e:
finally:
```
## enumerate()枚举类
```python
str1/list = "123"/[1,2,3]
for i,j in enumerate(str1/list1)
```
## 测试代码消耗时间
```python
import time
start = time.time()
#代码块
······
print(time.time()-start)
```
## 检查对象占用内存空间情况
```python
import sys
str1 = '123'
sys.getsizeof(str1)
```
## 字典合并
```python
dict1 = {}
dict2 = {}
combined_dice = {**dict1, **dict2}

dict1.update(dict2)
```
## 随机采样
```pyhon
import radom
str1 = "abcd"
n_samples = 3
random.sample(str1, n_samples)
```
## 检查唯一性
拿本身和Set（本身）比较大小 （len()）

