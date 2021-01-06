# MADE2_Face_Recognize
 NBUIT_MADE2_PROJECT

此项目是NBUT的人脸识别项目。目的是使学生更深入的了解线性代数，以及用线性代数及相关算法实现人脸识别应用。

## 使用的算法：PCA

大致过程是把人脸图像降维后转基得到人脸特征向量，把训练用的相同的人脸向量圈为一个区域，把待识别人脸向量与训练好的特征向量圈作欧氏距离对比，从而识别人脸。

## 转基分析：

在将图像降维拉伸后，如何选取一个一维线性空间，使得数据在其上的投影更分散，是算法的关键。

那么其本质是寻找一个一维基，使得所有数据变换为这个基上的坐标表示后，方差值最大。

我们先列出方差方程：
$$
S^2=\frac{1}{m-1}(V^TA)^2
$$
其中S是方差，m是样本量，V是单位基向量，A是样本矩阵，因为样本向量皆进行中心化处理，所以平均值为0，不用考虑。

上面方程可继续变换
$$
S^2=\frac{1}{m-1}(V^TA)^2\\
=\frac{1}{m-1}V^TA(V^TA)^T\\
=V^T(\frac{1}{m-1}AA^T)V
$$
其中$$\frac{1}{m-1}AA^T$$又是A的**协方差公式：**$$Cov(A)$$.

于是，原式继续化简
$$
S^2=V^T(\frac{1}{m-1}AA^T)V\\
=V^T\cdot Col(A)\cdot V
$$
可以看出，上式符合线性代数7.3节条件优化的定理6：

> 设A是对称矩阵，且m和M的定义符合
> $$
> m=\min\{x^TAx:\Vert x\Vert=1\},M=\max\{x^TAx:\Vert x\Vert =1\}
> $$
> 那么M是A的最大特征值$\lambda_1$,m是A的最小特征值。如果x是对应于M的单位特征向量u~1~，那么$x^TAx$的值等于M。如果x是对应于m的单位特征向量，那么$$x^TAx$$的值等于m。

于是，**Col(A)里的最大的特征值所对应的特征向量**就是我们要找的，使方差值最大的一维基。

## 程序实现：

### 1、图像拉伸降维

将图像转为灰度图像，即可用普通矩阵表示，然后将其拉伸为向量
$$
\begin{vmatrix}
 a_{11} & \cdots  & a_{1n}\\
 \vdots  & \ddots  & \vdots \\
 a_{n1} & \cdots & a_{nn}
\end{vmatrix}
=
\begin{vmatrix}
 a_{11} & \cdots  & a_{1n}  & \cdots  & a_{n1} & \cdots & a_{nn}
\end{vmatrix}
$$
然后将每个元素减去元素平均值，将向量中心化。

### 2、图像转基

- 整理出样本矩阵

  将m个图像降维成n维向量后，将它们组成$$n\times m$$矩阵A。

- 

将众多图像降维成n维向量后，为它们寻找一组新的，在R^n^里的标准正交基，使得某（几个）个基向量，对于图像集拥有较小的平均方差（图像向量在基向量上的投影较为密集）。

