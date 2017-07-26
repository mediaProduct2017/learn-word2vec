# learn-word2vec

## 1. Dot Product 

Dot product is a measure of similarity, euclidean distance is also a measure of similarity.

u<sup>T</sup>v = u * v = sum(u<sub>i</sub>v<sub>i</sub>)

Bigger if u and v are similar

## 2. Softmax function

p<sub>i</sub> = exp(u<sub>i</sub>)/sum(exp(u<sub>j</sub>))

u可能是线性函数（比如在multiple logistic regression中），也可能是简单的二次函数（比如坐标点乘）

p常用于分类问题（一般是多个类别）的forward propagation的概率计算，用来根据x得到分类y

## 3. Skip-gram

Skip-gram是一种用神经网络模型直接拟合词汇向量的方法，用来得到各个词的low dimentional vector。

word2vec算法所得到的词汇向量，用的就是skip-gram的方法。

Skip-gram拟合的核心是从中心词到周围词的预测，参数的选择以预测最准为目标。优化的cost function关注的就是从中心词到周围词的预测的概率，目标是实际情况的概率最大。

## 4. Cost function

Cost function是数据拟合中的优化目标，通过在back propagation中实现cost function的最优化，可以得到需要拟合的参数

## 5. Cross entropy cost function

两个概率分布之间的距离度量，常用于loss function或者cost function，特别是分类问题（一般是多个类别）的cost function

Kullback distance也可用于两个概率分布之间的距离度量

CE(y1i,y2i)=-sum(y1i*logy2i)

负号是为了将最大化问题转化为最小化问题

对于多分类问题，只有实际分类的logyi项被保留

一般情况下，yi就用softmax function的pi来计算，对于从中心词到周围词的预测（skip-gram拟合word方法），参数包括所有词的向量

在图像识别中，yi中的i表示不同的图片，yi表示具体的图像分类；对于从中心词到周围词的预测，yi中的i表示不同的位置，yi表示具体的词汇分类

## 6. Logistic cost function

-sum(y*log(hx)+(1-y)*log(1-hx))

y=1 or 0

在logistic regression或者neural network中，可用于二分类问题，也可用于多分类问题

## 7. Negative sampling cost function

log(sigma(uo<sup>T</sup>v)) + sum(log(sigma(-uj<sup>T</sup>v))

-sum(y*log(sigma(uo<sup>T</sup>v)) + (1-y)*log(sigma(-uj<sup>T</sup>v))

结果与logistic cost function有相似的地方

window中的某个位置的分类是某个词i，对这个词i来讲，yi=1

y=1: output word在window中
y=0: output word不在window中，是随机抽签的结果

参数包括y=1的词的向量，以及被随机抽中的y=0的词的向量，不像cross entropy cost function, 参数包括所有词的向量

## 8. Sigmoid function

logistic function 

p<sub>i</sub> = 1/(1+exp(-u<sub>i</sub>))
