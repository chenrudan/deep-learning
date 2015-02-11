This is a cplusplus code for the algorithm of CRBM.
---------------------------------------------------
CRBM的c++代码实现

代码内容包括：

白化类：用来处理输入数据的白化

卷积类：当输入数据，可进行卷积，输入两个矩阵，间隔

crbm结构类：包括filtering层，pooling层，向前卷积，向后卷积，计算权重

显示类：调用python接口来显示

矩阵计算类：加减乘除

文件名小写加下划线，变量小写加下划线，成员变量后面加上_，函数名首字母大写

在crbm中，输入图片存在vector<matrix>里面，权重也是，vector的大小就是channels的个数

在这个里面不用求cost，也就不用求freeenergy

dw有三个
