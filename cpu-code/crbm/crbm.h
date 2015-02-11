/*************************************************************************
    > File Name: crbm.h
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月14日 星期一 19时57分22秒
 ************************************************************************/
#ifndef CRBM_H
#define CRBM_H

#include <iostream>
#include <vector>
#include "matrix.h"

using namespace std;

class Crbm
{
    public:
        //每一层的结构
        Crbm();
        ~Crbm();

/* Function: FilterInit
 * --------------------
 * 这个函数初始化权重，输入接口为filter大小和通道数
 */
        void FilterInit(int filter_row, int num_channels, int input_channels, int input_row, int batch_size, int pooling_size);

/* Function: ConvolutionForward
 * ----------------------------
 * 从visbile得到Hidden，返回hidden的结果值，放在一个vector里面，这个结果是采样的结果，会使用到下一层进行pooling
 */
        void ConvolutionForward(vector<Matrix*> &input_image, int pos);

/* Function: ConvolutionBackward
 * ----------------------------
 * 从Hidden得到visbile，返回v的采样值，放在一个vector里面
 */
        void ConvolutionBackward(vector<Matrix*> &hidden_map);

/* Function: Sample
 * ----------------
 * 对输入的矩阵进行采样，返回采样结果
 */
        void Sample(Matrix *mat);

/* Function: ComputeDerivative
 * ---------------------
 * 更新权重，偏置
 */
        void ComputeDerivative(vector<Matrix*> &input_image, int pos);

/* Function: InitPars
 * ------------------
 * 当进行更新权重时初始化dw和pre_dw，当进行gibbs采样时初始化hn_sample等
 */
        void InitPars(vector<Matrix*> &pars, int channels);

/* Function: MaxPooling
 * --------------------
 * 对这一层的输出进行pooling
 */
        vector<Matrix*>* MaxPooling();
        int SubMaxPooling(float *prods, float sum);

/* Function: SupplyImage
 * --------------------
 * 将一张图按要求补零
 */
        Matrix* SupplyImage(Matrix* mat, int supply_size, bool is_supply_final);

/* Function: RunBatch
 * ------------------
 * 这个函数将整个过程串联起来，可调用它直接得到最后的pooling结果
 */
        vector<Matrix*>* RunBatch(vector<Matrix*> &input_image, int pos);

/* Function: GetWeight
 * -------------------
 * 返回权重值
 */
        vector<Matrix*>* GetWeight();
        vector<Matrix*>* GetPooling();

/* Function: GetReconstructionCost
 * --------------------------------
 * 返回重构损耗
 */
        float GetReconstructionCost(vector<Matrix*> &input_image, int pos);


    private:
        float l2reg_;
        float ph_lambda;
        float ph;
        //微小项
        float epsilon_;
        float momentum_;
        int CD_K_;
        int input_channels_;
        int num_channels_;
        int filter_row_;
        int input_row_;
        int pooling_size_;
        int batch_size_;
        //为true时表示求postive phase，false求negative
        bool first_conv_forward_;
        //暂时不知为何，在求输出时用到了，增大取值
        float std_gaussian_;
        //输出的图片大小
        int out_size_;
        //存放这一层的所有权重值
        vector<Matrix*> weight_;
        //存放这一层的输出图片
        vector<Matrix*> feature_map_;
        vector<Matrix*> unsample_feature_map_;
        vector<Matrix*> pooling_map_;
        //存放经过对比差异CD_K更新后的v
        vector<Matrix*> vn_sample_;
        vector<Matrix*> unsample_vn_sample_;
        //存放经过对比差异CD_K更新后的h
        vector<Matrix*> hn_sample_;
        vector<Matrix*> unsample_hn_sample_;
        //保存偏导
        vector<Matrix*> dw_;
        vector<Matrix*> pre_dw_;
        //对应channel
        vector<float> vbias_;
        vector<float> dvbias_;
        vector<float> pre_dvbias_;
        //对应bases
        vector<float> hbias_;
        vector<float> dhbias_;
        vector<float> pre_dhbias_;

};

#endif
