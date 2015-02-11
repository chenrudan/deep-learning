/*************************************************************************
    > File Name: matrix.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月13日 星期日 16时08分16秒
 ************************************************************************/

#include<iostream>
#include<vector>
#include"matrix.h"

using namespace std;

Matrix::Matrix(int row, int col)
{
    init(row, col);
}
Matrix::Matrix()
{

}

void Matrix::init(int row, int col)
{
    this->row_ = row;
	this->col_ = col;
	row_unfull_pos_ = 0;
	col_unfull_pos_ = 0;
	this->sum_ = 0;
	this->min_ = 100000;
	this->max_ = -100000;

	//为二维数组分配内存
	all_element_ = new float*[row];
	for(int i = 0; i < this->row_; i++)
	{
		all_element_[i] = new float[col];
		for(int j = 0; j < this->col_; j++)
		{
		    all_element_[i][j] = 0;
		}
	}
}

Matrix::~Matrix()
{
	for(int i = 0; i < this->row_; i++)
	{
		delete[] all_element_[i];
	}
	delete[] all_element_;

}

Matrix* Matrix::MatrixMultiply(Matrix *mat_1, Matrix *mat_2)
{
	long prod_row = mat_1->GetRowNum();
	long prod_col = mat_2->GetColNum();
	Matrix *prod_mat = new Matrix(prod_row, prod_col);
	for(long i = 0; i < prod_row; i++)
	{
		for(long j = 0; j < prod_col; j++)
		{
			for(long k = 0; k < mat_1->GetColNum(); k++)
			{
				prod_mat->all_element_[i][j] += mat_1->all_element_[i][k]*mat_2->all_element_[k][j];
			}
		}
	}
	return prod_mat;
}

void Matrix::MatrixMulCoef(float coef)
{
    for(int i = 0; i < this->row_; i++)
	{
		for(int j = 0; j < this->col_; j++)
		{
			this->all_element_[i][j] = coef*this->all_element_[i][j];
		}
	}
}

void Matrix::MatrixAddNew(Matrix *mat, float coef)
{
    if(this->row_ != mat->GetRowNum())
        cout << "the two matrix cannot do the addition!\n";
    else
    {
        for(long i = 0; i < this->row_; i++)
        {
            for(long j = 0; j < this->col_; j++)
            {
                this->all_element_[i][j] = this->all_element_[i][j] + coef*(mat->all_element_[i][j]);
            }
        }
    }
}

void Matrix::MatrixAssign(Matrix *mat, float coef)
{
    if(this->row_ != mat->GetRowNum())
        cout << "the matrix cannot do the assign!\n";
    else
    {
        for(long i = 0; i < this->row_; i++)
        {
            for(long j = 0; j < this->col_; j++)
            {
                this->all_element_[i][j] = coef*(mat->all_element_[i][j]);
            }
        }
    }
}

Matrix* Matrix::MatrixAdd(Matrix *mat_1, Matrix *mat_2)
{
    return MatrixAdd(mat_1, 1, mat_2, 1);
}

Matrix* Matrix::MatrixAdd(Matrix *mat_1, float coef_1, Matrix *mat_2, float coef_2)
{
    if((mat_1->GetRowNum() != mat_2->GetRowNum())||(mat_1->GetColNum() != mat_2->GetColNum()))
    {
        cout << "the two matrix cannot do the addition!\n";
    }
    else
    {
        long prod_row = mat_1->GetRowNum();
        long prod_col = mat_1->GetColNum();
        Matrix *prod_mat = new Matrix(prod_row, prod_col);
        for(long i = 0; i < prod_row; i++)
        {
            for(long j = 0; j < prod_col; j++)
            {
                prod_mat->all_element_[i][j] = coef_1*(mat_1->all_element_[i][j]) + coef_2*(mat_2->all_element_[i][j]);
            }
        }
        return prod_mat;
    }
    return 0;
}

Matrix* Matrix::MatrixSub(Matrix *mat_1, Matrix *mat_2)
{
    return MatrixAdd(mat_1, 1, mat_2, -1);
}
/*
Matrix* Matrix::MatrixSub(Matrix *mat_1, float coef_1, Matrix *mat_2, float coef_2)
{
    if((mat_1->GetRowNum() != mat_2->GetRowNum())||(mat_1->GetColNum() != mat_2->GetColNum()))
    {
        cout << "the two matrix cannot do the minus!\n";
    }
    else
    {
        long prod_row = mat_1->GetRowNum();
        long prod_col = mat_1->GetColNum();
        Matrix *prod_mat = new Matrix(prod_row, prod_col);
        for(long i = 0; i < prod_row; i++)
        {
            for(long j = 0; j < prod_col; j++)
            {
                prod_mat->all_element_[i][j] = coef_1*(mat_1->all_element_[i][j]) - coef_2*(mat_2->all_element_[i][j]);
            }
        }
        return prod_mat;
    }
    return 0;
}*/

Matrix* Matrix::MatrixAddBias(Matrix *mat_1, float bias)
{
    for(long i = 0; i < mat_1->GetRowNum(); i++)
    {
        for(long j = 0; j < mat_1->GetRowNum(); j++)
        {
            mat_1->all_element_[i][j] =  mat_1->all_element_[i][j] + bias;

        }
    }
    return mat_1;
}

Matrix* Matrix::MatrixTranspose()
{
    Matrix *transpose = new Matrix(this->row_, this->col_);
    for(int i = 0; i < this->row_; i++)
    {
        for(int j = 0; j < this->col_; j++)
        {
            transpose->ChangeElement(i, j, this->all_element_[j][i]);
        }
    }
    return transpose;
}


float Matrix::GetElement(long row, long col)
{
	return this->all_element_[row][col];
}

long Matrix::GetRowNum()
{
	return this->row_;
}

long Matrix::GetColNum()
{
	return this->col_;
}

void Matrix::AddElement(float element)
{
	all_element_[row_unfull_pos_][col_unfull_pos_] = element;
	col_unfull_pos_++;
	if(col_unfull_pos_ >= this->col_)
	{
		row_unfull_pos_++;
		col_unfull_pos_ = 0;
	}
	if(row_unfull_pos_ > this->row_)
	{
		cout << "you have enter two many element!\n" ;
		row_unfull_pos_--;
	}
	this->sum_ += element;
}

void Matrix::AddElementByCol(float value)
{
    all_element_[row_unfull_pos_][col_unfull_pos_] = value;
	row_unfull_pos_++;
	if(row_unfull_pos_ >= this->row_)
	{
		col_unfull_pos_++;
		row_unfull_pos_ = 0;
	}
	if(col_unfull_pos_ > this->col_)
	{
		cout << "you have enter two many element!\n" ;
		row_unfull_pos_--;
	}
}

void Matrix::ChangeElement(long row, long col, float value)
{
    all_element_[row][col] = value;
}


float Matrix::MatrixSum()
{
    for(int i = 0; i < this->row_; i++)
    {
        for(int j = 0; j < this->col_; j++)
        {
            sum_ =+ all_element_[i][j];
        }
    }
    return this->sum_;
}

float Matrix::MatrixMin()
{
    for(int i = 0; i < this->row_; i++)
    {
        for(int j = 0; j < this->col_; j++)
        {
             if(this->min_ > all_element_[i][j])
                this->min_ = all_element_[i][j];
        }
    }
    return this->min_;
}

float Matrix::MatrixMax()
{
    for(int i = 0; i < this->row_; i++)
    {
        for(int j = 0; j < this->col_; j++)
        {
             if(this->max_ < all_element_[i][j])
                this->max_ = all_element_[i][j];
        }
    }
    return this->max_;
}

float Matrix::MatrixSum(Matrix *mat)
{
    return mat->MatrixSum();
}

float Matrix::MatrixAverage()
{
    return (this->sum_/(this->row_*this->col_));
}

float Matrix::MatrixAverage(Matrix *mat)
{
    return mat->MatrixAverage();
}

void Matrix::ClearElement()
{
    for(int i = 0; i < this->row_; i++)
	{
		delete[] all_element_[i];
	}
	delete[] all_element_;

}

void Matrix::Display()
{
    for(int i = 0; i < this->row_; i++)
    {
        for(int j = 0; j < this->col_; j++)
        {
            cout << this->all_element_[i][j] << endl;
        }
    }
}






























































