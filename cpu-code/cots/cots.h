/*************************************************************************
    > File Name: cots.h
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com 
    > Created Time: 2014年08月30日 星期六 20时26分05秒
 ************************************************************************/

#ifndef COTS_H
#define COTS_H

#include<iostream>

using namespace std;

class Cots
{
	public:
		Cots();
        Cots(int input_size, int input_channels, int filter_size, int filter_channels, int batch_size, int block_size, \
                int step, int process_num, int thread_num, int pooling_size, float learning_rate, float alpha, float learning_rate_alpha, float momentum, float lambda, string address);
		~Cots();
		void init(int input_size, int input_channels, int filter_size, int filter_channels, int batch_size, int block_size, \
                int step, int process_num, int thread_num, int pooling_size, float learning_rate, float alpha, float learning_rate_alpha, float momentum, float lambda, string address);
/* Function: trainModel
 * --------------------
 * 给外界提供接口进行训练，外界提供me和epoch，和整个训练集个数,type原来是指是否进行预处理，现在认为每一层的输入都要进行预处理
 */
		void testModel(int me, int epoch, int batch_all_size, bool type);
        void trainModel(int me, int epoch, int batch_all_size, bool type);
        void preprocess(int batch_idx, int num, int channels, int size, float *input, bool type);
/* Function: initWeight
 * --------------------
 * 初始化weight
 */
		void initWeight(int me,  bool type);
		
/* Function: filterLayer
 * ---------------------
 * filter层
 */
 		void filterLayer(int me, int batch_idx);

/* Function: filterLayer
 * ---------------------
 * filter层
 */
        void lcnLayer(int me, int batch_idx);

/* Function: poolingLayer
 * ---------------------
 * pooling层
 */		
		void poolingLayer(int me, int batch_idx);
 		
/* Function: computeH
 * -------------------
 * 用来计算w*x
 */
		void computeH(int process_idx);

/* Function: buildH
 * ----------------
 * 将h还原成大的h，或者将p，lcn还原成全部的p，lcn
 */
		void buildH(int me, float *block, float *all, int process_idx, bool ward);
		
/* Function: computeR
 * ------------------
 * 计算每一个小的r
 */
		void computeR(int process_idx);
		
/* Function: buildR
 * ----------------
 * 将r还原成大的r
 */
		void buildR(int me, float *block, float *all, int process_idx, bool ward);

/* Function: normalizeWeight
 * -------------------------
 * 模变为1
 */ 		
		void normalizeWeight();
        void inverseProjWeight(float *graident, float *origin_weight, float *project_weight);
/* Function: computeP
 * ------------------
 * 计算pooling层的值，每次计算2*2的区域，取值则取到3*3，hidden的平方和开方，同时算出dw2
 */
		void computeP(int me, int process_idx, int type);
		
/* Function: computeLcn
 * ------------------
 * 计算lcn层的值，每次计算2*2的区域，取值则取到3*3
 */
		void computeLcn(int me, int process_idx, bool type);
		
/* Function: updateW
 * -------------------
 * 更新权重
 */
		void updateW(int me, int batch_idx);
		void updateAlpha();
/* Function: computeDw1
 * --------------------
 * 计算fliter层造成的偏导，h*(r-x)'+ w*(r-x)*x'
 */
	    void computeDw1(float *block_dw1, int process_idx);
		
/* Function: computeDw2
 * --------------------
 * 计算pooling层造成的偏导
 */
 		void computeDw2(float *block_dw2);

/* Function: saveFile
 * ------------------
 * 保存weight，r，h
 */
        void subSaveFile(string filename, int length, float *data, bool type);

        void zeros(float *all, int length);
/* Function: assignMemory
 * ----------------------
 * 分配内存
 */
		void assignMemory();
		
/* Function: clearMemory
 * ----------------------
 * 删除之前分配的内存
 */
		void clearMemory();
		
	private:
		int _me;
		int _epoch;
        float _lambda;
        float _delta_alpha;
        int _block_sqrt;
        int _input_sqrt;
        int _out_sqrt;
        int _filter_sqrt;
        int _pooling_sqrt;
		string _address;
		struct _Pars{
			int input_channels;
			int input_size;
			int filter_size;
			int filter_channels;
			int batch_size;
			int block_size;
			int step;
			int process_num;
            int thread_num;
			int out_size;
			int pooling_size;
			float learning_rate;
			float learning_rate_alpha;
			float alpha;
			float momentum;

			//zero pass to other thread
			float *block_input;
			//weight for block compute
			float *block_weight;
			float *block_hidden;
			float *block_reconstruct;
			float *block_pooling;
			float *block_lcn;
			
			float *input;
			//r for zero
			float *send_reconstruct;
			float *receive_reconstruct;
			//hidden for zero
			float *send_hidden;
			float *receive_hidden;
			//pooling 
			float *send_pooling;
			float *receive_pooling;
			//lcn
			float *send_lcn;
			float *receive_lcn;
			//weight for zero
			float *weight;
			float *winc;
			float *normalize_weight;
		};
        struct _Pars *_pars;
};

#endif












