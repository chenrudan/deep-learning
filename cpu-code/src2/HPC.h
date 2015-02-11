#ifndef STL10_H
#define STL10_H

#include<iostream>
#include<vector>
using namespace std;

class HPC{

public:
    void free();
    void malloc();
    void free_maximization();
    void malloc_maximization();
    HPC& set_minibatch(int mminibatch);
    HPC& set_alpha(float aalpha);
    HPC& set_ela(float eela);
    HPC& set_theata(float ttheata);
    HPC& set_c(float cc);
    HPC& set_learning_rate(float llearning_rate);
    HPC& set_lcn_channel(int llcn_channel);
    HPC& set_lcn_row(int llcn_row);
    HPC& set_lcn_col(int llcn_col);
    HPC& set_inputFileName(string iinputFileName);
    HPC& set_h_filename(string hh_filename);
    HPC& set_p_filename(string pp_filename);
    HPC& set_lcn_filename(string llcn_filename);
    HPC& set_picture_number(int ppicture_number);
    float* get_pixelD() const;
    float* get_rpixelD() const;
    float* get_pixelD_block() const;
    float* get_rpixelD_block() const;
    float* get_rpixelD_gather() const;
    float* get_h() const;
    float* get_w() const;
    float* get_n_w() const;
    float* get_w_gather() const;
    float* get_delta_w() const;
    float* get_delta_w_m() const;
    float* get_h_gather() const;
    float* get_h_sort() const;
    float* get_p() const;
    float* get_lcn() const;
    float* get_h_delta_alpha() const;
    float* get_h_delta_w() const;
    float get_alpha() const;
    int get_minibatch() const;
    void zeros_rpixelD();
    void zeros_rpixelD_gather();
    void zeros_delta_w();
    void zeros_w_gather();
    void preprocess();
    void loaddata_Integer(int minibatch_index);
    void loaddata_Float(int minibatch_index);
    void loaddata_minist(int minibatch_index);
    void loaddata_cifar_100(int minibatch_index);
    void initlize_w();
    void normalize_w();
    void loaddata_w(string w_filename);
    void loaddata_w(const float* ww);
    void initlize_pixelD_block(int index);
    void initlize_rpixelD_block(int index,int flag);
    void initlize_h_block(int index);
    void initlize_lcn_GW();
    void initlize_delta_w_m();
    void store_rpixel_block(int block_index);
    void sort_h_gather();
    void pooling();
    void Lcn();
    void write_h();
    void write_p();
    void write_lcn();

    void compute_rpixelD();
    void compute_h();
    void copy_rpixelD();
    void compute_delta_w();
    void compute_h_delta_alpha();
    void compute_h_delta_w();
    float compute_delta_alpha2();
    float compute_delta_alpha1();
    
    //comput the gradient with input_x;
    //h->hidden layer output
    //p->pooling layer output
    //v->lcn layer output
    void copy_h_to_h_gather();
    void copy_w_gather_to_w();
    void compute_delta_hx();
    void compute_delta_ph();
    void compute_delta_vp();
    void compute_delta_lv();

    void compute_delta_px();
    void compute_delta_vx();
    void compute_delta_lx();

    void get_delta_hx_block(int i);
    void get_delta_px_block(int i);
    void get_delta_vx_block(int i);

    float* get_delta_hx() const;
    float* get_delta_vx() const;
    float* get_delta_px() const;
    float* get_delta_lx() const;
    float* get_lc() const;
    float* get_max_x() const;

    void random_x();
    void normalize_x();
    void projected_gradient(float* delta,float* ori,float* n_ori,int size);
    void copy_to_x(float* pixel,int size);

    HPC(int rrank,int pprocess_number,char *pprocessor_name,int iinput_row,int iinput_col,int iinput_channel,string iinputFileName,int ppicture_number,int ffilter_row,int ffilter_col,int sstep_size,int hhidden_channel,int hhidden_row,int hhidden_col,int ppooling_row,int ppooling_col);
    HPC(int iinput_row,int iinput_col,int iinput_channel,string iinputFileName,int ppicture_number,int ffilter_row,int ffilter_col,int sstep_size,int hhidden_channel,int hhidden_row,int hhidden_col,int ppooling_row,int ppooling_col);

    //the three channel images are stored in one vector ,order by R G B ,each image is col major
    //as is to say ,the first image pixels is R channel and col major
private:
    int rank;
    int process_number;
    char *processor_name;
    
    //input layer
    int input_row;
    int input_col;
    int input_channel;
    string inputFileName;
    int picture_number;

    //filter
    int filter_row;
    int filter_col;
    int step_size;

    //hiddenlayer
    int hidden_block_row;
    int hidden_block_col;
    int hidden_channel;
    int hidden_row;
    int hidden_col;
    string h_filename;

    //pooling layer
    int pooling_row;
    int pooling_col;
    float *pooling_weight;
    string p_filename;

    //LCN layer
    int lcn_channel; 
    int lcn_row;
    int lcn_col;
    float *lcn_GW;
    float *lcn;
    string lcn_filename;

    //model relative
    int minibatch;
    //rpixelD=alpha*W^T*H
    float alpha;
    //use for projected grident
    float theata;
    //use for lcn
    float c;
    //ela*(H/p).X^T
    float ela;
    float learning_rate;
    int hiddenBlocksNumberOfPerProcess;
    float *pixelD;
    float *rpixelD;
    float *pixelD_block;
    float *rpixelD_block;
    float *rpixelD_x_block;
    float *rpixelD_gather;
    float *w;
    float *n_w;
    float *w_gather;
    float *h;
    float *h_gather;
    float *h_sort;
    float *h_delta_alpha;
    float *h_delta_w;
    float *h_block;
    float *p;
    float *delta_w;
    float *delta_w_m;

    //maximization
    float *delta_hx;
    float *delta_ph;
    float *delta_vp;
    float *delta_lv;

    float *delta_hx_block;
    float *delta_px_block;
    float *delta_vx_block;

    float *delta_px;
    float *delta_vx;
    float *delta_lx;
    
    float *max_x;
    float *lc;
    float *ln1;
    float *ln2;
};
#endif

