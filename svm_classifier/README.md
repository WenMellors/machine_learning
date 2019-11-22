## 特征观察

#### X1, X3, X10, X11, X12 比率类型

能够对样本在坐标系上的绝对位置进行标定 -> 空间标点？

计算比率类型数据的距离就用欧式距离这些就可

相似度计算：$s = -d$ 或 $s = 1 - \frac{d - minD}{maxD - minD}$ 

#### x4 序数类型

取值范围为 1-16

序数类型能够对样本之间的顺序进行区分，如排名、年级、衣服的尺寸。（聚类？）

序数类型变量的保序变换是等价变换

序数类型的距离：$d = \frac{|p-q|}{n-1}$ (一共 n 种序数)

对应相似度计算：$s= 1 - d$

#### x2, x5, x6, x7, x8, x9 标称类型

取值范围都很小

只能区分样本之间的不同

任何一对一变换是等价变换

标称类型的距离：海明距离，相同为 1，不同为 0

#### label

样本中负样本有 19752，正样本有 6272 个