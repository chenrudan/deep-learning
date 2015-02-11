/*************************************************************************
    > File Name: crbm.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月14日 星期一 19时57分29秒
 ************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <cstring>
#include "crbm.h"
#include "matrix.h"
#include "utils.h"
#include "conv.h"
#include "show.h"

using namespace std;

Crbm::Crbm()
{
    this->CD_K_ = 1;
 //   this->weight_ = new vector<Matrix*>();
 //   this->feature_map_ = new vector<Matrix*>();
    this->first_conv_forward_ = true;
    this->l2reg_ = 0.01;
    this->ph_lambda = 5;
    this->ph = 0.002;
    this->epsilon_ = 0.005;
    this->momentum_ = 0.5;
    this->pooling_size_ = 2;
    this->std_gaussian_ = 0.2;
}

Crbm::~Crbm()
{
}

void Crbm::FilterInit(int filter_row, int num_channels, int input_channels, \
                      int input_row, int batch_size, int pooling_size)
{
    this->num_channels_ = num_channels;
    this->filter_row_ = filter_row;
    this->input_channels_ = input_channels;
    this->input_row_ = input_row;
    this->out_size_ = this->input_row_ - this->filter_row_ + 1;
    this->batch_size_ = batch_size;
    this->pooling_size_ = pooling_size;

//    float high = 4 * sqrt(6.0 / (2 * filter_row * filter_row * num_channels));
 //   float low = -high;
    for(int i = 0; i < num_channels*input_channels; i++)
    {
        Matrix *tmp_weight = new Matrix(filter_row_, filter_row_);
        Matrix *tmp_dw = new Matrix(filter_row_, filter_row_);
        Matrix *tmp_pre_dw = new Matrix(filter_row_, filter_row_);
        this->dw_.push_back(tmp_dw);
        this->pre_dw_.push_back(tmp_pre_dw);
        //初始化权重
        for(int j = 0; j < filter_row * filter_row - 1; j++)
        {
            float rand_value = 0.01*RandomNormal();
            tmp_weight->ChangeElement(j/filter_row_, j%filter_row_, rand_value);
        }
        this->weight_.push_back(tmp_weight);
    }
    //初始化vn_sample
    for(int i = 0; i < this->batch_size_; i++)
    {
        Matrix *tmp_vn_sample = new Matrix[input_channels_];
        Matrix *tmp_umsample_vn_sample = new Matrix[input_channels_];
        for(int j = 0; j < this->input_channels_; j++)
        {
            tmp_vn_sample[j].init(this->input_row_, this->input_row_);
            tmp_umsample_vn_sample[j].init(this->input_row_, this->input_row_);
        }
        vn_sample_.push_back(tmp_vn_sample);
        unsample_vn_sample_.push_back(tmp_umsample_vn_sample);
    }

    for(int j = 0; j < this->input_channels_; j++)
    {
        //初始化v偏置
        this->vbias_.push_back(0.0);
        this->dvbias_.push_back(0.0);
        this->pre_dvbias_.push_back(0.0);
    }
    //初始化hn_sample

    for(int i = 0; i < this->batch_size_; i++)
    {
        Matrix *tmp_hn_sample = new Matrix[num_channels_];
        Matrix *tmp_feature_map = new Matrix[num_channels_];
        Matrix *tmp_unsample = new Matrix[num_channels_];
        Matrix *tmp_unsample_hn_sample = new Matrix[num_channels_];
        for(int j = 0; j < this->num_channels_; j++)
        {
            tmp_hn_sample[j].init(this->out_size_, this->out_size_);
            tmp_feature_map[j].init(this->out_size_, this->out_size_);
            tmp_unsample[j].init(this->out_size_, this->out_size_);
            tmp_unsample_hn_sample[j].init(this->out_size_, this->out_size_);
        }
        hn_sample_.push_back(tmp_hn_sample);
        //初始化feature_map
        feature_map_.push_back(tmp_feature_map);
        unsample_feature_map_.push_back(tmp_unsample);
        unsample_hn_sample_.push_back(tmp_unsample_hn_sample);

    }
    for(int i = 0; i < this->num_channels_; i++)
    {
        //初始化h偏置
        this->hbias_.push_back(-0.1);
        this->dhbias_.push_back(0.0);
        this->pre_dhbias_.push_back(0.0);
    }
}

//vector<Matrix*>* Crbm::ConvolutionForward(vector<Matrix*> *input_image)
void Crbm::ConvolutionForward(vector<Matrix*> &input_image, int pos)
{
    if(this->first_conv_forward_)
    {
        InitPars(unsample_feature_map_, num_channels_);
        InitPars(feature_map_, num_channels_);
        for(int batch_idx = 0; batch_idx < this->batch_size_; batch_idx++)
        {
            for(int i = 0, k = 0; i < this->num_channels_; i++)
            {
                for(int j = 0; j < this->input_channels_; j++, k++)
                {
                    (unsample_feature_map_.at(batch_idx) + i)->MatrixAddNew(  \
                                     Conv::Conv2d(input_image.at(batch_idx + pos) + j, weight_.at(k), 1), 1);
                     //       (input_image.at(batch_idx + pos) + j)->Display();
                }
                Matrix::MatrixAddBias((unsample_feature_map_.at(batch_idx) + i), hbias_[i]);

                (unsample_feature_map_.at(batch_idx) + i)->MatrixMulCoef(1.0/(std_gaussian_*std_gaussian_));
             //   (unsample_feature_map_.at(batch_idx) + i)->Display();
          //      Show show;
           //     show.ShowMyMatrix8U(unsample_feature_map_.at(batch_idx) + i);
          /*      for(int m = 0; m < this->out_size_; m++)
                {
                    for(int n = 0; n < this->out_size_; n++)
                    {
                        (feature_map_.at(batch_idx) + i)->ChangeElement(m, n, (unsample_feature_map_.at(batch_idx) + i)->GetElement(m, n));
                    }
                }*/
                //采样得到h1
       //         Sample(feature_map_.at(batch_idx) +i);
            }
        }
        this->first_conv_forward_ = false;
    }
    else
    {
        InitPars(unsample_hn_sample_, num_channels_);
        //当进行gibbs采样时
        for(int batch_idx = 0; batch_idx < this->batch_size_; batch_idx++)
        {
            for(int i = 0, k = 0; i < this->num_channels_; i++)
            {
                for(int j = 0; j < this->input_channels_; j++, k++)
                {
                    (unsample_hn_sample_.at(batch_idx) + i)->MatrixAddNew( \
                                    Conv::Conv2d((input_image.at(batch_idx + pos) + j), weight_.at(k), 1) , 1);
                }
                Matrix::MatrixAddBias((unsample_hn_sample_.at(batch_idx) + i), hbias_[i]);
                (unsample_hn_sample_.at(batch_idx) + i)->MatrixMulCoef(1.0/(std_gaussian_*std_gaussian_));
             //   (unsample_hn_sample_.at(batch_idx) + i)->Display();
                for(int m = 0; m < this->out_size_/pooling_size_; m ++)
                {
                    for(int n = 0; n < this->out_size_/pooling_size_; n ++)
                    {
                        //针对四个块进行处理
                        float probs[this->pooling_size_*this->pooling_size_];
                        float sum = 0;
                        for(int row = 0; row < this->pooling_size_; row++)
                        {
                            for(int col = 0; col < this->pooling_size_; col++)
                            {
                                probs[row*this->pooling_size_ + col] = Logisitc( \
                                    (unsample_hn_sample_.at(batch_idx) + i)->GetElement(m*pooling_size_+row, n*pooling_size_+col));
                                sum += probs[row*this->pooling_size_ + col];
                            }
                        }
                        for(int row = 0; row < this->pooling_size_; row++)
                        {
                            for(int col = 0; col < this->pooling_size_; col++)
                            {
                                (unsample_hn_sample_.at(batch_idx) + i)->ChangeElement(m*pooling_size_+row, \
                                                                n*pooling_size_+col, probs[row*this->pooling_size_ + col]/(1+sum));
                            }
                        }
                    }
                }
              //  (unsample_hn_sample_.at(batch_idx) + i)->Display();
            }
        }
    }
}

void Crbm::ConvolutionBackward(vector<Matrix*> &hidden_sample)
{
    InitPars(unsample_vn_sample_, input_channels_);
    for(int batch_idx = 0; batch_idx < this->batch_size_; batch_idx++)
    {
        for(int i = 0; i < this->num_channels_; i++)
        {
            Matrix *supply_hidden;
            supply_hidden = SupplyImage(hidden_sample.at(batch_idx) + i, filter_row_ - 1, false);
           // Show show;
           // show.ShowMyMatrix8U(supply_hidden);
          //  supply_hidden->Display();
           // (weight_.at(i*(this->input_channels_)))->Display();
            //与三个权重做卷积
            for(int j = 0 ; j < this->input_channels_; j++)
            {
                (unsample_vn_sample_.at(batch_idx) + j)->MatrixAddNew( \
                            Conv::Conv2d(supply_hidden, (weight_.at(i*(this->input_channels_) + j)->MatrixTranspose()), 1), 1);
            }

            supply_hidden->ClearElement();
        }

        for(int i = 0; i < input_channels_; i++)
        {
              Matrix::MatrixAddBias((unsample_vn_sample_.at(batch_idx) + i), vbias_[i]);

         //   Show show;
         //   show.ShowMyMatrix8U(unsample_vn_sample_.at(batch_idx) + i);
         //    (unsample_vn_sample_.at(batch_idx) + i)->MatrixMulCoef(1/24);
        //   (unsample_vn_sample_.at(batch_idx) + i)->Display();
      /*      for(int m = 0; m < this->input_row_; m++)
            {
                for(int n = 0; n < this->input_row_; n++)
                {
                    (vn_sample_.at(batch_idx) + i)->ChangeElement(m, n, (unsample_vn_sample_.at(batch_idx) + i)->GetElement(m, n));
                }
            }
            Sample(vn_sample_.at(batch_idx) + i);*/
      //      (vn_sample_.at(batch_idx) + i)->Display();
        }
    }
}

void Crbm::InitPars(vector<Matrix*> &paras, int channels)
{
    int length = paras.size();
    int para_size = paras.at(0)->GetRowNum();
    //数字全部置为0就可
    for(int i = 0; i < length; i++)
    {
        for(int j = 0; j < channels; j++)
        {
            for(int m = 0; m < para_size; m++)
            {
                for(int n = 0; n < para_size; n++)
                {
                    paras[i][j].ChangeElement(m, n, 0.0);
                }
            }
        }
    }
}

void Crbm::Sample(Matrix *mat)
{
/*    float mean;
    for(int i = 0; i < mat->GetRowNum(); i++)
    {
        for(int j = 0; j < mat->GetColNum(); j++)
        {
            mean = Logisitc(mat->GetElement(i, j));
            if(mean < 0 || mean > 1)
                mat->ChangeElement(i, j, 0.0);
            else
            {
                if(CompareFloat(mean, RandomNumber()))
                    mat->ChangeElement(i, j, 1.0);
                else
                    mat->ChangeElement(i, j, 0.0);
            }
        }
    }*/
}

void Crbm::ComputeDerivative(vector<Matrix*> &input_image, int pos)
{
    //一张输入图和一张输出图来更新一组w，w共24*3组
    InitPars(this->dw_, 1);
    //InitPars(this->pre_dw_, 1);
    for(int i = 0; i < this->num_channels_; i++)
    {
        for(int j = 0; j < this->input_channels_; j++)
        {
            for(int batch_idx = 0; batch_idx < this->batch_size_; batch_idx++)
            {
                Matrix* tmp_dw;
                tmp_dw = Matrix::MatrixAdd( Conv::Conv2d((input_image.at(batch_idx + pos) + j), \
                                                         unsample_feature_map_.at(batch_idx) + i, 1), 1, \
                                             Conv::Conv2d((unsample_vn_sample_.at(batch_idx) + j), \
                                                          unsample_hn_sample_.at(batch_idx) + i, 1), -1);

                dw_.at(i*input_channels_ + j)->MatrixAddNew(tmp_dw, 1);
                tmp_dw->ClearElement();
            }
            //weight_.at(i*input_channels_ + j)->Display();
            dw_.at(i*input_channels_ + j)->MatrixMulCoef(1.0/(this->batch_size_*this->out_size_*this->out_size_));
            dw_.at(i*input_channels_ + j)->MatrixAddNew(weight_.at(i*input_channels_ + j), -l2reg_);
            dw_.at(i*input_channels_ + j)->MatrixMulCoef(this->epsilon_);
            dw_.at(i*input_channels_ + j)->MatrixAddNew(pre_dw_.at(i*input_channels_ + j), momentum_);
            //将dw的内容拷贝到pre_dw
            for(int m = 0; m < this->filter_row_; m++)
            {
                for(int n = 0; n < this->filter_row_; n++)
                {
                    pre_dw_.at(i*input_channels_ + j)->ChangeElement(m, n, dw_.at(i*input_channels_ + j)->GetElement(m, n));
                }
            }

            //更新w
            this->weight_.at(i*input_channels_ + j)->MatrixAddNew(this->dw_.at(i*input_channels_ + j), 1);
       //     weight_.at(i*input_channels_ + j)->Display();
        }
        //计算dh
        dhbias_.at(i) = 0;
        for(int batch_idx = 0; batch_idx < this->batch_size_; batch_idx++)
        {
            this->dhbias_.at(i) += ((unsample_feature_map_.at(batch_idx) + i)->MatrixSum() - \
                                    (unsample_hn_sample_.at(batch_idx) + i)->MatrixSum())  \
                                    /(this->out_size_ * this->out_size_) - \
                                    this->ph_lambda * ((unsample_feature_map_.at(batch_idx) + i)->MatrixAverage() - this->ph);
        }
        this->dhbias_.at(i) = (this->epsilon_*this->dhbias_.at(i) + this->momentum_*this->pre_dhbias_.at(i))   \
                                    /(this->batch_size_*this->out_size_*this->out_size_);
        this->pre_dhbias_.at(i) = this->dhbias_.at(i);
        //更新hbias
        this->hbias_.at(i) = this->hbias_.at(i) + this->dhbias_.at(i);

    }
    for(int i = 0; i < this->input_channels_; i++)
    {
        this->dvbias_.at(i) = 0;
        this->dvbias_.at(i) = this->epsilon_*this->dvbias_.at(i) + this->momentum_*this->pre_dvbias_.at(i);
        this->pre_dvbias_.at(i) = this->dvbias_.at(i);
        //更新vbias
        this->vbias_.at(i) = this->vbias_.at(i) + this->dvbias_.at(i);
    }
}

vector<Matrix*>* Crbm::MaxPooling()
{
    if(this->out_size_%pooling_size_)
    {
        for(int batch_idx = 0; batch_idx < this->batch_size_; batch_idx++)
        {
            for(int i = 0; i < this->num_channels_; i++)
            {
                 SupplyImage(unsample_feature_map_.at(batch_idx) + i, this->out_size_%pooling_size_, true);
            }
        }
        this->out_size_ += this->out_size_%pooling_size_;
    }
    for(int batch_idx = 0; batch_idx < this->batch_size_; batch_idx++)
    {
        Matrix *p_pooling = new Matrix[num_channels_];
        for(int i = 0; i < this->num_channels_; i++)
        {
            p_pooling[i].init(out_size_/pooling_size_, out_size_/pooling_size_);
            for(int m = 0; m < this->out_size_/pooling_size_; m ++)
            {
                for(int n = 0; n < this->out_size_/pooling_size_; n ++)
                {
                    //针对四个块进行处理
                    float probs[this->pooling_size_*this->pooling_size_ + 1];
                    float sum = 0;
                    for(int row = 0; row < this->pooling_size_; row++)
                    {
                        for(int col = 0; col < this->pooling_size_; col++)
                        {
                            probs[row*this->pooling_size_ + col] = Logisitc(\
                                    (unsample_feature_map_.at(batch_idx) + i)->GetElement(m*pooling_size_+row, n*pooling_size_+col));
                            sum += probs[row*this->pooling_size_ + col];
                        }
                    }
                    for(int row = 0; row < this->pooling_size_; row++)
                    {
                        for(int col = 0; col < this->pooling_size_; col++)
                        {
                            (unsample_feature_map_.at(batch_idx) + i)->ChangeElement(\
                                         m*pooling_size_+row, n*pooling_size_+col, probs[row*this->pooling_size_ + col]/(1+sum));
                        }
                    }

                    probs[this->pooling_size_*this->pooling_size_] = 1 / (1 + sum);
                    int pos = SubMaxPooling(probs, sum);
                    if(pos >= pooling_size_*pooling_size_)
                    {
                        p_pooling[i].AddElement(0.0);
                    }
                    else
                    {
                        p_pooling[i].AddElement(1.0);
                        (feature_map_.at(batch_idx) + i)->ChangeElement(\
                                        m*pooling_size_+pos/pooling_size_, n*pooling_size_+pos%pooling_size_, 1);
                    }
                }
            }
           //        Show show;
           // show.ShowMyMatrix8U(unsample_feature_map_.at(batch_idx) + i);
          //(unsample_feature_map_.at(batch_idx) + i)->Display();
          //(feature_map_.at(batch_idx) + i)->Display();
            pooling_map_.push_back(p_pooling);
        }

    }
    return &pooling_map_;
}

vector<Matrix*>* Crbm::GetPooling()
{
    return &this->pooling_map_;
}


int Crbm::SubMaxPooling(float *probs, float sum)
{
    for(int row = 0; row < this->pooling_size_; row++)
    {
        for(int col = 0; col < this->pooling_size_; col++)
        {
            probs[row*this->pooling_size_ + col] = probs[row*this->pooling_size_ + col] / (1 + sum);
        }
    }
    float t = RandomNumber();
    int i;
    for(i = 0; t > probs[i]; i++, probs[i] += probs[i-1]);
    return i;
}

Matrix* Crbm::SupplyImage(Matrix* mat, int supply_size, bool is_supply_final)
{
    if(!is_supply_final)
    {
        int size = mat->GetRowNum() + 2*supply_size;
        Matrix *new_mat = new Matrix(size, size);
        for(int m = supply_size; m < size - supply_size ; m++)
        {
            for(int n = supply_size; n < size - supply_size ; n++)
            {
                new_mat->ChangeElement(m, n, mat->GetElement(m - supply_size, n - supply_size));
            }
        }
        return new_mat;
    }
    else
    {
        int size = mat->GetRowNum() + supply_size;
        Matrix *new_mat = new Matrix(size, size);
        for(int m = 0; m < size - supply_size; m++)
        {
            for(int n = 0; n < size - supply_size; n++)
            {
                new_mat->ChangeElement(m, n, mat->GetElement(m, n));
            }
        }
        mat->ClearElement();
        memcpy(mat, new_mat, sizeof(Matrix));
        return new_mat;
    }
}


vector<Matrix*>* Crbm::RunBatch(vector<Matrix*> &input_image, int pos)
{
    ConvolutionForward(input_image, pos);
    cout << "postive phase , forward success!" << endl;
    MaxPooling();
    cout << "pooling success!" << endl;
    //3.开始contrastive divergence
    for(int i = 0; i < this->CD_K_; i++)
    {
        if(i == 0)
        {
            ConvolutionBackward(this->feature_map_);
        }
        else
            ConvolutionBackward(this->hn_sample_);
        ConvolutionForward(this->unsample_vn_sample_, 0);
    }
 //   Show show;
 //   show.ShowMyMatrix8U(unsample_vn_sample_.at(0));
    cout << "contrastive divergence success!\n";
    //4.更新权重
    ComputeDerivative(input_image, pos);
    cout << "ComputeDerivative success!\n";
    first_conv_forward_ = true;
    cout << "cross cost is :" << GetReconstructionCost(input_image, pos) << endl;
    //5.max_pooling并输出结果
    return GetPooling();
}

vector<Matrix*>* Crbm::GetWeight()
{
    return &this->weight_;
}

float Crbm::GetReconstructionCost(vector<Matrix*> &input_image, int pos)
{
    float sum = 0.0;
    for(int batch_idx = 0; batch_idx < this->batch_size_; batch_idx++)
    {
        for(int i = 0; i < this->input_channels_; i++)
        {
            for(int m = 0; m < this->input_row_; m++)
            {
                for(int n = 0; n < this->input_row_; n++)
                {
                    sum += ((input_image.at(batch_idx + pos) + i)->GetElement(m,n)  \
                            - (unsample_vn_sample_.at(batch_idx) + i)->GetElement(m,n))   \
                                *((input_image.at(batch_idx + pos) + i)->GetElement(m,n)  \
                            - (unsample_vn_sample_.at(batch_idx) + i)->GetElement(m,n));
                }
            }
        }
    }
    return sqrt(sum);
    //(batch_size_*input_channels_*input_row_*input_row_);
}























