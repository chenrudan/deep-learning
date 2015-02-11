/*************************************************************************
    > File Name: matrix.h
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月12日 星期六 20时25分15秒
 ************************************************************************/
#ifndef MATRIX_H
#define MATRIX_H

#include<iostream>
#include<vector>

using namespace std;

class Matrix
{
	public:
		Matrix(int row, int col);
		Matrix();
		~Matrix();

		void init(int row, int col);

/* Function: MatrixMultiply
 * ------------------------
 * 做矩阵乘法，输入两个矩阵，得到返回值矩阵的指针
 */
		static Matrix* MatrixMultiply(Matrix *mat_1, Matrix *mat_2);
		void MatrixMulCoef(float coef);

/* Function: MatrixAdd
 * ------------------------
 * 做矩阵加法，输入两个矩阵，得到返回值矩阵的指针
 */
        static Matrix* MatrixAdd(Matrix *mat_1, Matrix *mat_2);
        void MatrixAddNew(Matrix *mat, float coef);
        static Matrix* MatrixAdd(Matrix *mat_1, float coef_1, Matrix *mat_2, float coef_2);
        void MatrixAssign(Matrix *mat, float coef);


/* Function: MatrixSub
 * ------------------------
 * 做矩阵减法，输入两个矩阵，得到返回值矩阵的指针
 */
        static Matrix* MatrixSub(Matrix *mat_1, Matrix *mat_2);
        static Matrix* MatrixSub(Matrix *mat_1, float coef_1, Matrix *mat_2, float coef_2);

/* Function: MatrixAddBias
 * ------------------------
 * 做矩阵加法，输入两个矩阵，得到返回值矩阵的指针
 */
        static Matrix* MatrixAddBias(Matrix *mat_1, float bias);

/* Function: MatrixTranspose
 * ------------------------
 * 做矩阵转置
 */
        Matrix* MatrixTranspose();

/* Function: MatrixSum
 * -----------------------
 * 返回整个矩阵的总和
 */
        float MatrixSum();
        float MatrixMin();
        float MatrixMax();
        float MatrixSum(Matrix *mat);

 /* Function: MatrixAverage
  * -----------------------
  * 返回整个矩阵的均值
  */
        float MatrixAverage();
        float MatrixAverage(Matrix *mat);

/* Function: AddElement
 * -------------------
 * 向矩阵添加元素
 */
		void AddElement(float value);
		void AddElementByCol(float value);

/* Function: GetRowNum
 * -------------------
 * 返回行
 */
		long GetRowNum();

/* Function: GetColNum
 * -------------------
 * 返回行
 */
		long GetColNum();

/* Function: GetElement
 * ---------------------
 * 返回输入的行列对应元素
 */
		float GetElement(long row, long col);

/* Function: ChangeElement
 * ---------------------
 * 返回输入的行列对应元素
 */
        void ChangeElement(long row, long col, float value);

/* Function: ClearElement
 * ----------------------
 * 删除内部的元素值
 */
        void ClearElement();
        void Display();


	private:
		//记录行列值
		long row_;
		long col_;
		//记录哪行已经填满
		long row_unfull_pos_;
		long col_unfull_pos_;
		//一列为一张图片的像素值，或者多张图片的像素值，一行为batchSize
		//假如表示权重，则对应该图片的权重值得到hidden层的图片
		float **all_element_;
		float sum_;
		float min_;
		float max_;
};

#endif /*matrix.h*/
