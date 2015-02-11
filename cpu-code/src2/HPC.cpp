#include"HPC.h"
#include"math.h"
#include"memory.h"
#include<fstream>
extern "C"{
#include"cblas.h"
#include<stdlib.h>
}

HPC::HPC(int rrank,int pprocess_number,char *pprocessor_name,int iinput_row,int iinput_col,int iinput_channel,string iinputFileName,int ppicture_number,int ffilter_row,int ffilter_col,int sstep_size,int hhidden_block_row,int hhidden_block_col,int hhidden_channel,int ppooling_row,int ppooling_col){

    //mpi 相关
    rank=rrank;
    process_number=pprocess_number;
    processor_name=pprocessor_name;
    //input layer
    input_row=iinput_row;
    input_col=iinput_col;
    input_channel=iinput_channel;
    inputFileName=iinputFileName;
    picture_number=ppicture_number;
    //filter
    filter_row=ffilter_row;
    filter_col=ffilter_col;
    step_size=sstep_size;
    //hidden layer
    hidden_block_row=hhidden_block_row;
    hidden_block_col=hhidden_block_col;
    hidden_channel=hhidden_channel;
    hidden_row=((input_row-filter_row)/step_size+1)*hidden_block_row;
    hidden_col=((input_col-filter_col)/step_size+1)*hidden_block_col;
    //pooling layer
    pooling_row=ppooling_row;
    pooling_col=ppooling_col;

    //model
    hiddenBlocksNumberOfPerProcess=(hidden_row/hidden_block_row)*(hidden_col/hidden_block_col)/process_number;
}

HPC::HPC(int iinput_row,int iinput_col,int iinput_channel,string iinputFileName,int ppicture_number,int ffilter_row,int ffilter_col,int sstep_size,int hhidden_block_row,int hhidden_block_col,int hhidden_channel,int ppooling_row,int ppooling_col){

    rank=0;
    process_number=1; 
    //input layer
    input_row=iinput_row;
    input_col=iinput_col;
    input_channel=iinput_channel;
    inputFileName=iinputFileName;
    picture_number=ppicture_number;
    //filter
    filter_row=ffilter_row;
    filter_col=ffilter_col;
    step_size=sstep_size;
    //hidden layer
    hidden_block_row=hhidden_block_row;
    hidden_block_col=hhidden_block_col;
    hidden_channel=hhidden_channel;
    hidden_row=((input_row-filter_row)/step_size+1)*hidden_block_row;
    hidden_col=((input_col-filter_col)/step_size+1)*hidden_block_col;
    //pooling layer
    pooling_row=ppooling_row;
    pooling_col=ppooling_col;

    //model
    hiddenBlocksNumberOfPerProcess=(hidden_row/hidden_block_row)*(hidden_col/hidden_block_col)/process_number;
}
void HPC::malloc(){
    int size_pixelD=input_row*input_col*input_channel*minibatch;
    int size_pixelD_block=filter_row*filter_col*input_channel*minibatch;
    int size_w=hiddenBlocksNumberOfPerProcess*filter_row*filter_col*input_channel*hidden_block_row*hidden_block_col*hidden_channel;
    int size_h=hidden_block_row*hidden_block_col*hidden_channel*hiddenBlocksNumberOfPerProcess*minibatch;
    int size_h_gather=hidden_row*hidden_col*hidden_channel*minibatch;
    //int size_p=(hidden_row-pooling_row+1)*(hidden_col-pooling_col+1);
    int size_p=size_h_gather;

    pixelD=new float[size_pixelD];
    rpixelD=new float[size_pixelD];
    rpixelD_gather=new float[size_pixelD];
    pixelD_block=new float[size_pixelD_block];
    rpixelD_block=new float[size_pixelD_block];
    rpixelD_x_block=new float[size_pixelD_block];
    w=new float[size_w];
    n_w=new float[size_w];
    w_gather=new float[size_w*process_number];
    delta_w=new float[size_w];
    delta_w_m=new float[size_w];
    h=new float[size_h];
    h_gather=new float[size_h_gather];
    h_sort=new float[size_h_gather];
    h_delta_alpha=new float[size_h_gather];
    h_delta_w=new float[size_h_gather];
    h_block=new float[size_h/hiddenBlocksNumberOfPerProcess];
    p=new float[size_p];
    lcn_GW=new float[lcn_row*lcn_col];
    lcn=new float[size_p];
    lc=new float[size_p];
    ln1=new float[size_p/hidden_channel];
    ln2=new float[size_p/hidden_channel];
}

void HPC::malloc_maximization(){
    int size_pixelD=input_row*input_col*input_channel*minibatch;
    int size_pixelD_block=filter_row*filter_col*input_channel*minibatch;
    int size_w=hiddenBlocksNumberOfPerProcess*filter_row*filter_col*input_channel*hidden_block_row*hidden_block_col*hidden_channel;
    int size_h=hidden_block_row*hidden_block_col*hidden_channel*hiddenBlocksNumberOfPerProcess*minibatch;
    int size_h_gather=hidden_row*hidden_col*hidden_channel*minibatch;
    //int size_p=(hidden_row-pooling_row+1)*(hidden_col-pooling_col+1);
    int size_p=size_h_gather;

    pixelD=new float[size_pixelD];
    pixelD_block=new float[size_pixelD_block];
    w=new float[size_w];
    w_gather=new float[size_w*process_number];
    n_w=new float[size_w];
    h=new float[size_h];
    h_gather=new float[size_h_gather];
    h_sort=new float[size_h_gather];
    p=new float[size_p];
    lcn_GW=new float[lcn_row*lcn_col];
    lcn=new float[size_p];
    lc=new float[size_p];
    ln1=new float[size_p/hidden_channel];
    ln2=new float[size_p/hidden_channel];

    int size_delta_hx=size_h_gather*size_pixelD;
    int size_delta_ph=size_h_gather*pooling_row*pooling_col;
    int size_delta_vp=size_h_gather*lcn_row*lcn_col*hidden_channel;
    int size_delta_hx_block=pooling_row*pooling_col*size_pixelD;
    int size_delta_px_block=lcn_row*lcn_col*hidden_channel*size_pixelD;

    max_x=new float[input_row*input_col*input_channel];
    delta_hx=new float[size_delta_hx];
    delta_ph=new float[size_delta_ph];
    delta_vp=new float[size_delta_vp];
    delta_lv=new float[size_delta_vp];

    delta_px=new float[size_delta_hx];
    delta_vx=new float[size_delta_hx];
    delta_lx=new float[size_delta_hx];

    delta_hx_block=new float[size_delta_hx_block];
    delta_px_block=new float[size_delta_px_block];
    delta_vx_block=new float[size_delta_px_block];

    memset(delta_hx,0,sizeof(float)*size_delta_hx);
    memset(delta_ph,0,sizeof(float)*size_delta_ph);
    memset(delta_vp,0,sizeof(float)*size_delta_vp);
    memset(delta_lv,0,sizeof(float)*size_delta_vp);

    memset(delta_px,0,sizeof(float)*size_delta_hx);
    memset(delta_vx,0,sizeof(float)*size_delta_hx);
    memset(delta_lx,0,sizeof(float)*size_delta_hx);
    memset(delta_hx_block,0,sizeof(float)*size_delta_hx_block);
    memset(delta_px_block,0,sizeof(float)*size_delta_px_block);
    memset(delta_vx_block,0,sizeof(float)*size_delta_px_block);
}

void HPC::free(){
    delete[] pixelD;
    delete[] rpixelD;
    delete[] rpixelD_gather;
    delete[] pixelD_block;
    delete[] rpixelD_block;
    delete[] rpixelD_x_block;
    delete[] w;
    delete[] n_w;
    delete[] w_gather;
    delete[] delta_w;
    delete[] delta_w_m;
    delete[] h;
    delete[] h_gather;
    delete[] h_sort;
    delete[] h_delta_alpha;
    delete[] h_delta_w;
    delete[] h_block;
    delete[] p;
    delete[] lcn_GW;
    delete[] lcn;
    delete[] lc;
    delete[] ln1;
    delete[] ln2;
}

void HPC::free_maximization(){
    delete[] pixelD;
    delete[] pixelD_block;
    delete[] w;
    delete[] w_gather;
    delete[] n_w;
    delete[] h;
    delete[] h_gather;
    delete[] h_sort;
    delete[] p;
    delete[] lcn_GW;
    delete[] lcn;
    delete[] lc;
    delete[] ln1;
    delete[] ln2;
    delete[] max_x;
    delete[] delta_hx;
    delete[] delta_ph;
    delete[] delta_vp;
    delete[] delta_px;
    delete[] delta_vx;
    delete[] delta_hx_block;
    delete[] delta_px_block;
    delete[] delta_vx_block;
}

//load binary data ,which store the integer between 0~255, just cost  one byte
void HPC::loaddata_Integer(int minibatch_index){
    int size_input=input_row*input_col*input_channel*minibatch;

    ifstream fp(inputFileName.c_str(),ios::in|ios::binary);
    char* buffer;
    buffer=new char[size_input];
    fp.seekg(minibatch_index*size_input,ios::beg);
    fp.read(buffer,size_input);
    fp.close();

    for(int j=0;j<size_input;j++){
        unsigned char a=buffer[j];
        pixelD[j]=a/255.0;
    }
    delete[] buffer;
}

void HPC::loaddata_Float(int minibatch_index){
    int size_input=input_row*input_col*input_channel*minibatch;
    ifstream fp(inputFileName.c_str(),ios::in|ios::binary);
    fp.seekg(minibatch_index*size_input*sizeof(float),ios::beg);
    fp.read((char*)pixelD,sizeof(float)*size_input);
    fp.close();
}

void HPC::loaddata_minist(int minibatch_index){
    int size_input=input_row*input_col*input_channel*minibatch;
    ifstream fp(inputFileName.c_str(),ios::in|ios::binary);
    char* buffer;
    buffer=new char[size_input];
    fp.seekg(minibatch_index*size_input+16,ios::beg);
    fp.read(buffer,size_input);
    fp.close();
    for(int j=0;j<size_input;j++){
        unsigned char a=buffer[j];
        pixelD[j]=a/255.0;
    }
    delete[] buffer;
}

void HPC::loaddata_cifar_100(int minibatch_index){
    int size_input=input_row*input_col*input_channel;
    ifstream fp(inputFileName.c_str(),ios::in|ios::binary);
    long begin_index=(2+size_input)*minibatch*minibatch_index;
    fp.seekg(begin_index,ios::beg);
    char* buffer;
    buffer=new char[size_input+2];
    for(int i=0;i<minibatch;i++){
        fp.read(buffer,size_input+2);
        for(int j=0;j<size_input;j++){
            unsigned char a=buffer[j+2];
            pixelD[i*size_input+j]=a/255.0;
        }
    }
    fp.close();
    delete[] buffer;
}

void HPC::random_x(){
    int size_max_x=input_row*input_col*input_channel;
    for(int i=0;i<size_max_x;i++){
        int a=rand()%200;
        //max_x[i]=a/100.0;
        pixelD[i]=(a-100.0)/100.0;
    }
}
void HPC::copy_to_x(float* pixel,int size){
    for(int i=0;i<size;i++){
        pixelD[i]=pixel[i];
    }
}

void HPC::normalize_x(){
    for(int i=0;i<input_channel;i++){
        int index=i*input_row*input_col;
        float sum=0.0;
        for(int j=0;j<input_row*input_col;j++){
            //sum+=max_x[index+j]*max_x[index+j];
            sum+=pixelD[index+j]*pixelD[index+j];
        }
        sum=sqrt(sum);
        for(int j=0;j<input_row*input_col;j++){
            pixelD[index+j]=pixelD[index+j]/sum;
        }
    }

}

//whitening the pictures of a minibatch
void HPC::preprocess(){
    for(int i=0;i<minibatch*input_channel;i++){
        int index=i*input_row*input_col;
        float mean=0.0;
        for(int j=0;j<input_row*input_col;j++){
            mean+=pixelD[index+j];
        }
        mean=mean/(input_row*input_col);
        for(int j=0;j<input_row*input_col;j++){
            pixelD[index+j]=pixelD[index+j]-mean;
        }

        float sum=0.0;
        for(int j=0;j<input_row*input_col;j++){
            sum+=pixelD[index+j]*pixelD[index+j];
        }
        sum=sqrt(sum);
        for(int j=0;j<input_row*input_col;j++){
            pixelD[index+j]=pixelD[index+j]/sum;
        }
    }
}

void HPC::initlize_w(){
    int size_w=hiddenBlocksNumberOfPerProcess*filter_row*filter_col*input_channel*hidden_block_row*hidden_block_col*hidden_channel;
    for(int i=0;i<size_w;i++){
        int a=rand()%2000;
        w[i]=(a-1000)/1000.0;
    }
}

void HPC::normalize_w(){
    int numberOfFilters=hiddenBlocksNumberOfPerProcess*hidden_block_row*hidden_block_col*hidden_channel;
    int size_filter=filter_row*filter_col*input_channel;
    for(int i=0;i<numberOfFilters;i++){
        float sum=theata;
        for(int j=0;j<size_filter;j++){
            sum+=w[i*size_filter+j]*w[i*size_filter+j];
        }
        sum=sqrt(sum);
        for(int j=0;j<size_filter;j++){
            n_w[i*size_filter+j]=w[i*size_filter+j]/sum;
        }
    }
}

void HPC::loaddata_w(string w_filename){
    int size_w_gather=hidden_block_row*hidden_block_col*hidden_channel*filter_row*filter_col*input_channel*hiddenBlocksNumberOfPerProcess*process_number;
    ifstream fp(w_filename.c_str(),ios::in|ios::binary);
    fp.read((char*)w_gather,sizeof(float)*size_w_gather);
    fp.close();
}

void HPC::loaddata_w(const float* ww){
    int size_w=hidden_block_row*hidden_block_col*hidden_channel*filter_row*filter_col*input_channel*hiddenBlocksNumberOfPerProcess;
    for(int i=0;i<size_w;i++){
        w[i]=ww[i];
    }
}

void HPC::initlize_pixelD_block(int index){
    int block_id=rank*hiddenBlocksNumberOfPerProcess+index;
    int block_row=block_id%(hidden_row/hidden_block_row);
    int block_col=block_id/(hidden_row/hidden_block_row);

    int size_sigle_channel=input_row*input_col;
    int size_sigle_picture=size_sigle_channel*input_channel;
    int index_pixelD_block=0;

    for(int i=0;i<input_channel;i++){
        for(int j=0;j<filter_row*filter_col;j++){
            for(int m=0;m<minibatch;m++){
                int col=j/filter_row;
                int row=j%filter_row;
                col=block_col*step_size+col;
                row=block_row*step_size+row;
                pixelD_block[index_pixelD_block++]=pixelD[i*size_sigle_channel+m*size_sigle_picture+col*input_row+row];
            }
        }
    }
}

//if flag=0 get rpixelD_block from rpixelD,so as to get rpixelD
//if flag=1 get rpixelD_x_block from rpixelD_gather,so as to get (rpixelD-pixelD)
void HPC::initlize_rpixelD_block(int index,int flag){
    int block_id=rank*hiddenBlocksNumberOfPerProcess+index;
    int block_row=block_id%(hidden_row/hidden_block_row);
    int block_col=block_id/(hidden_row/hidden_block_row);

    int size_sigle_channel=input_row*input_col;
    int size_sigle_picture=size_sigle_channel*input_channel;
    int index_pixelD_block=0;

    for(int i=0;i<input_channel;i++){
        for(int j=0;j<filter_row*filter_col;j++){
            for(int m=0;m<minibatch;m++){
                int col=j/filter_row;
                int row=j%filter_row;
                col=block_col*step_size+col;
                row=block_row*step_size+row;
                if(flag==0) rpixelD_block[index_pixelD_block++]=rpixelD[i*size_sigle_channel+m*size_sigle_picture+col*input_row+row];
                else rpixelD_x_block[index_pixelD_block++]=rpixelD_gather[i*size_sigle_channel+m*size_sigle_picture+col*input_row+row];
            }
        }
    }

}

void HPC::initlize_h_block(int index){
    int flag=0;
    int index_block=index+rank*hiddenBlocksNumberOfPerProcess;
    int block_col=index_block/(hidden_row/hidden_block_row);
    int block_row=index_block%(hidden_row/hidden_block_row);

    int size_sigle_picture=hidden_row*hidden_col*hidden_channel;
    int size_sigle_channel=hidden_row*hidden_col;
    for(int i=0;i<hidden_channel;i++){
        for(int col=0;col<hidden_block_col;col++){
            for(int row=0;row<hidden_block_row;row++){
                for(int j=0;j<minibatch;j++){
                    int real_row=block_row*hidden_block_row+row;
                    int real_col=block_col*hidden_block_col+col;
                    h_block[flag++]=h_delta_w[j*size_sigle_picture+i*size_sigle_channel+real_col*hidden_row+real_row];
                }
            }
        }
    }
}

void HPC::initlize_lcn_GW(){
    lcn_GW[0]=0.0625/hidden_channel;
    lcn_GW[1]=0.125/hidden_channel;
    lcn_GW[2]=0.0625/hidden_channel;
    lcn_GW[3]=0.125/hidden_channel;
    lcn_GW[4]=0.25/hidden_channel;
    lcn_GW[5]=0.125/hidden_channel;
    lcn_GW[6]=0.0625/hidden_channel;
    lcn_GW[7]=0.125/hidden_channel;
    lcn_GW[8]=0.0625/hidden_channel;
}

void HPC::initlize_delta_w_m(){
    int size_w=hidden_block_row*hidden_block_col*hidden_channel*filter_row*filter_col*input_channel*hiddenBlocksNumberOfPerProcess;
    for(int i=0;i<size_w;i++){
        delta_w_m[i]=delta_w[i];
    }
}

void HPC::store_rpixel_block(int block_index){
    int block_id=rank*hiddenBlocksNumberOfPerProcess+block_index;
    int block_row=block_id%(hidden_row/hidden_block_row);
    int block_col=block_id/(hidden_row/hidden_block_row);

    int size_sigle_channel=input_row*input_col;
    int size_sigle_picture=size_sigle_channel*input_channel;
    int index_rpixelD_block=0;

    for(int i=0;i<input_channel;i++){
        for(int j=0;j<filter_row*filter_col;j++){
            for(int m=0;m<minibatch;m++){
                int col=j/filter_row;
                int row=j%filter_row;
                col=block_col*step_size+col;
                row=block_row*step_size+row;
                rpixelD[i*size_sigle_channel+m*size_sigle_picture+col*input_row+row]+=rpixelD_block[index_rpixelD_block++];
            }
        }
    }

}

void HPC::copy_h_to_h_gather(){
    int size_h=hidden_block_row*hidden_block_col*hidden_channel*hiddenBlocksNumberOfPerProcess*minibatch;
    for(int i=0;i<size_h;i++){
        h_gather[i]=h[i];
    }
}

void HPC::copy_w_gather_to_w(){
    int size_w=hidden_block_row*hidden_block_col*hidden_channel*filter_row*filter_col*input_channel*hiddenBlocksNumberOfPerProcess;
    for(int i=0;i<size_w;i++){
        w[i]=w_gather[i];
    }
}

void HPC::sort_h_gather(){
    int size_h_block=hidden_block_col*hidden_block_row*hidden_channel*minibatch;
    int size_h_block_sigle_channel=hidden_block_col*hidden_block_row*minibatch;
    int index=0;
    for(int i=0;i<minibatch;i++){
        for(int j=0;j<hidden_channel;j++){
            for(int n=0;n<hidden_col;n++){
                for(int m=0;m<hidden_row;m++){
                    int block_id=(n/hidden_block_col)*(hidden_row/hidden_block_row)+m/hidden_block_row;
                    int index_block_row=m%hidden_block_row;
                    int index_block_col=n%hidden_block_col;
                    h_sort[index++]=h_gather[block_id*size_h_block+j*size_h_block_sigle_channel+index_block_col*hidden_block_row*minibatch+index_block_row*minibatch+i];
                }
            }
        }
    }
}

void HPC::pooling(){
    int size_p=hidden_row*hidden_col*hidden_channel*minibatch;
    for(int i=0;i<size_p;i++){
        int index_picture=i/(hidden_row*hidden_col);
        int index_xy=i%(hidden_row*hidden_col);
        int row=index_xy%hidden_row;
        int col=index_xy/hidden_row;
        float sum=0;
        int a[]={-1,-1,-1,0,-1,1,0,-1,0,0,0,1,1,-1,1,0,1,1};
        for(int j=0;j<9;j++){
            int new_row=row+a[2*j];
            int new_col=col+a[2*j+1];
            if(new_row>=0&&new_row<hidden_row&&new_col>=0&&new_col<hidden_col){
                int index=new_row+new_col*hidden_row+index_picture*hidden_row*hidden_col;
                sum+=h_sort[index]*h_sort[index];
            }
        }
        p[i]=sqrt(sum);
    }
}

void HPC::Lcn(){
    int a[]={-1,-1,-1,0,-1,1,0,-1,0,0,0,1,1,-1,1,0,1,1};
    int size_lcn=hidden_row*hidden_col;
    int size_sigle_picture=hidden_row*hidden_col*hidden_channel;
    float *lcn_1=new float[size_lcn];
    for(int m=0;m<minibatch;m++){
        for(int i=0;i<size_lcn;i++){
            int row=i%hidden_row;
            int col=i/hidden_row;
            float sum=0;
            for(int j=0;j<9;j++){
                int new_row=row+a[2*j];
                int new_col=col+a[2*j+1];
                if(new_row>=0&&new_row<hidden_row&&new_col>=0&&new_col<hidden_col){
                    int index=new_row+new_col*hidden_row;
                    for(int k=0;k<hidden_channel;k++){
                        int index_p=index+k*size_lcn+m*size_sigle_picture;
                        sum+=p[index_p]*lcn_GW[j];
                    }
                }
            }
            lcn_1[i]=sum;
        }

        for(int i=0;i<size_sigle_picture;i++){
            //lc[m*size_sigle_picture+i]=p[m*size_sigle_picture+i]-lcn_1[i%size_lcn];
            lcn[m*size_sigle_picture+i]=p[m*size_sigle_picture+i]-lcn_1[i%size_lcn];
        }
        /*
        for(int i=0;i<size_lcn;i++){
            int row=i%hidden_row;
            int col=i/hidden_row;
            float sum=0;
            for(int j=0;j<9;j++){
                int new_row=row+a[2*j];
                int new_col=col+a[2*j+1];
                if(new_row>=0&&new_row<hidden_row&&new_col>=0&&new_col<hidden_col){
                    int index=new_row+new_col*hidden_row;
                    for(int k=0;k<hidden_channel;k++){
                        sum+=lc[m*size_sigle_picture+index+k*size_lcn]*lc[m*size_sigle_picture+index+k*size_lcn]*lcn_GW[j];
                    }
                }
            }
            //ln1 use for compute delta_lcn/delta_p
            //ln1[m*size_lcn+i]=sum1;
            ln2[m*size_lcn+i]=sqrt(sum);
        }*/
        /*
           float mean=0;
           float sum_lcn_1=0;
           for(int i=0;i<size_lcn;i++){
           sum_lcn_1+=lcn_1[i];
           }
           mean=sum_lcn_1/size_lcn;
           cout<<"mean_lcn:"<<mean<<"\t"; 
           *
           * /
    
        for(int i=0;i<size_sigle_picture;i++){
            float max=c>ln2[m*size_lcn+i%size_lcn]?c:ln2[m*size_lcn+i%size_lcn];
            //float max=ln2[m*size_lcn+i%size_lcn];
            //float max=c;
            lcn[i+m*size_sigle_picture]=lc[m*size_sigle_picture+i]/max;
        }*/
    }
    delete[] lcn_1;
}

void HPC::compute_h_delta_w(){
    int size_p=hidden_row*hidden_col*hidden_channel*minibatch;
    for(int i=0;i<size_p;i++){
        int index_picture=i/(hidden_row*hidden_col);
        int index_xy=i%(hidden_row*hidden_col);
        int row=index_xy%hidden_row;
        int col=index_xy/hidden_row;
        float sum=0;
        int a[]={-1,-1,-1,0,-1,1,0,-1,0,0,0,1,1,-1,1,0,1,1};
        for(int j=0;j<9;j++){
            int new_row=row+a[2*j];
            int new_col=col+a[2*j+1];
            if(new_row>=0&&new_row<hidden_row&&new_col>=0&&new_col<hidden_col){
                int index=new_row+new_col*hidden_row+index_picture*hidden_row*hidden_col;
                sum+=h_sort[i]/p[index];
            }
        }
        h_delta_w[i]=sum;
    }
}

//not use now,computing delta_alpha2 just using p
void HPC::compute_h_delta_alpha(){
    int size_p=hidden_row*hidden_col*hidden_channel*minibatch;
    for(int i=0;i<size_p;i++){
        int index_picture=i/(hidden_row*hidden_col);
        int index_xy=i%(hidden_row*hidden_col);
        int row=index_xy%hidden_row;
        int col=index_xy/hidden_row;
        float sum=0;
        int a[]={-1,-1,-1,0,-1,1,0,-1,0,0,0,1,1,-1,1,0,1,1};
        for(int j=0;j<9;j++){
            int new_row=row+a[2*j];
            int new_col=col+a[2*j+1];
            if(new_row>=0&&new_row<hidden_row&&new_col>=0&&new_col<hidden_col){
                int index=new_row+new_col*hidden_row+index_picture*hidden_row*hidden_col;
                sum+=h_sort[i]*h_sort[i]/p[index];
            }
        }
        h_delta_alpha[i]=sum;
    }
}

void HPC::compute_rpixelD(){
    int size_w_block=hidden_block_row*hidden_block_col*hidden_channel*filter_row*filter_col*input_channel;
    int size_h_block=hidden_block_row*hidden_block_col*hidden_channel*minibatch;
    int M=hidden_block_row*hidden_block_col*hidden_channel;
    int K=filter_row*filter_col*input_channel;
    int N=minibatch;
    for(int i=0;i<hiddenBlocksNumberOfPerProcess;i++){
        initlize_pixelD_block(i);
        //计算W[M*K]* X[K*N]= H[M*N]
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,alpha,&n_w[i*size_w_block],K,pixelD_block,N,0,&h[i*size_h_block],N);
        //计算W^T[K*M]*H[M*N]=R[K*N]
        cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,K,N,M,1,&n_w[i*size_w_block],K,&h[i*size_h_block],N,0,rpixelD_block,N);
        store_rpixel_block(i);
    }
}

void HPC::compute_h(){
    int size_w_block=hidden_block_row*hidden_block_col*hidden_channel*filter_row*filter_col*input_channel;
    int size_h_block=hidden_block_row*hidden_block_col*hidden_channel*minibatch;
    int M=hidden_block_row*hidden_block_col*hidden_channel;
    int K=filter_row*filter_col*input_channel;
    int N=minibatch;
    for(int i=0;i<hiddenBlocksNumberOfPerProcess;i++){
        initlize_pixelD_block(i);
        //计算W[M*K]* X[K*N]= H[M*N]
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,alpha,&n_w[i*size_w_block],K,pixelD_block,N,0,&h[i*size_h_block],N);
    }
}

void HPC::compute_delta_w(){
    const int M=hidden_block_row*hidden_block_col*hidden_channel;
    const int K=minibatch;
    const int N=filter_row*filter_col*input_channel;
    int size_h_block=hidden_block_row*hidden_block_col*hidden_channel*minibatch;
    int size_w_block=hidden_block_row*hidden_block_col*hidden_channel*filter_row*filter_col*input_channel;
    int size_wi=filter_row*filter_col*input_channel;

    for(int i=0;i<hiddenBlocksNumberOfPerProcess;i++){
        //从pixelD中获取该rank对应的一个block的X，存入pixelD_block中,comput delta_w and delta_alpha
        initlize_pixelD_block(i);
        //从rpixelD_gather中获取该rank对应的一个block的（R-X），存入rpixelD_x_block中,compute delta_w
        initlize_rpixelD_block(i,1);
        //从rpixelD中获取该rank 对应的一个block存入rpixelD_block中,comput delta_alpha
        initlize_rpixelD_block(i,0);

        //从h_sort中获取该rank 对应的一个block的(H/P)，存入h_block中
        initlize_h_block(i);
        //计算H(M*K)*(R_X)^T存入delta_w中
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,M,N,K,2.0/minibatch,&h[i*size_h_block],K,rpixelD_x_block,K,0,&delta_w[i*size_w_block],N);
        //计算W(R-X)存入h中,将h中当前block对应的值覆盖掉
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,K,N,1,&n_w[i*size_w_block],N,rpixelD_x_block,K,0,&h[i*size_h_block],K);
        //计算W(R-X)X^T,即H＊X^T存入delta_w当中，与原有的H(R_X)^T的结果相加，即得到了每个block中的delta_w 
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,M,N,K,2.0*alpha/minibatch,&h[i*size_h_block],K,pixelD_block,K,1,&delta_w[i*size_w_block],N);
        if(rank==0&&i==0){
            cout<<"---------delta_w1------------"<<endl;
            for(int j=0;j<100;j++){
                cout<<delta_w[j]<<" ";
            }
            cout<<endl;
        }
        //compute ela*(H/P)*X^T
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,M,N,K,ela*alpha/minibatch,h_block,K,pixelD_block,K,1,&delta_w[i*size_w_block],N);
        if(rank==0&&i==0){
            cout<<"---------delta_w1+delta_w2------------"<<endl;
            for(int j=0;j<100;j++){
                cout<<delta_w[j]<<" ";
            }
            cout<<endl;
        }

        //projected gradient methods
        for(int j=0;j<hidden_block_row*hidden_block_col*hidden_channel;j++){
            float tem=theata;
            float tem1=0;
            for(int k=0;k<size_wi;k++){
                int index=i*size_w_block+j*size_wi+k;
                tem+=w[index]*w[index];
                tem1+=delta_w[index]*w[index];
            }
            for(int k=0;k<size_wi;k++){
                int index=i*size_w_block+j*size_wi+k;
                delta_w[index]=delta_w[index]/sqrt(tem)-n_w[index]*tem1/tem;
            }

        }
    }

}

void HPC::projected_gradient(float* delta,float* ori,float* n_ori,int size){
    float tem=theata;
    float tem1=0;
    for(int i=0;i<size;i++){
        tem+=max_x[i]*max_x[i];
        tem1+=max_x[i]*delta[i];
    }
    for(int i=0;i<size;i++){
        delta[i]=delta[i]/sqrt(tem)-pixelD[i]*tem1/tem;
    }
}
//sparsty cost gradient with alpha
float HPC::compute_delta_alpha2(){
    int size_p=hidden_row*hidden_col*hidden_channel*minibatch;
    float delta_alpha=0;
    for(int i=0;i<size_p;i++){
        delta_alpha+=p[i];
    }
    delta_alpha=delta_alpha*ela/alpha;
    delta_alpha=delta_alpha/minibatch;
    return delta_alpha;
}

//Reconstruct cost gradient with alpha 
float HPC::compute_delta_alpha1(){
    int size_pixelD=input_row*input_col*input_channel*minibatch;
    float delta_alpha=0;
    for(int i=0;i<size_pixelD;i++){
        delta_alpha+=rpixelD[i]*rpixelD_gather[i];
    }
    delta_alpha=delta_alpha*2/alpha;
    delta_alpha=delta_alpha/minibatch;
    return delta_alpha;
}

void HPC::compute_delta_hx(){
    int p=0;
    for(int i=0;i<hiddenBlocksNumberOfPerProcess;i++){
        int block_row=i%(hidden_row/hidden_block_row);
        int block_col=i/(hidden_row/hidden_block_row);
        for(int j=0;j<hidden_block_row*hidden_block_col*hidden_channel;j++){
            int row_channel=j/(hidden_block_row*hidden_block_col);
            int row_row=block_row*step_size+(j%(hidden_block_row*hidden_block_col))%hidden_block_row;
            int row_col=block_col*step_size+(j%(hidden_block_row*hidden_block_col))/hidden_block_row;
            int row_index=row_channel*hidden_row*hidden_col+row_col*hidden_row+row_row;
            for(int k=0;k<filter_row*filter_col*input_channel;k++){
                int col_channel=k/(filter_row*filter_col);
                int col_row=block_row*step_size+(k%(filter_row*filter_col))%filter_row;
                int col_col=block_col*step_size+(k%(filter_row*filter_col))/filter_row;
                int col_index=col_channel*input_row*input_col+col_col*input_row+col_row;
                int index_hx=row_index*input_row*input_col*input_channel+col_index;
                delta_hx[index_hx]=w[p]*alpha;
                p++;
            }
        }
    }
}

void HPC::compute_delta_ph(){
    int size_h_gather=hidden_row*hidden_col*hidden_channel*minibatch;
    int size_p=hidden_row*hidden_col*hidden_channel*minibatch;
    int size_pooling=pooling_row*pooling_col;
    int size_delta_ph=size_h_gather*size_pooling;
    memset(delta_ph,0,sizeof(float)*size_delta_ph);
    for(int i=0;i<size_p;i++){
        int index_picture=i/(hidden_row*hidden_col);
        int index_xy=i%(hidden_row*hidden_col);
        int row=index_xy%hidden_row;
        int col=index_xy/hidden_row;
        int a[]={-1,-1,-1,0,-1,1,0,-1,0,0,0,1,1,-1,1,0,1,1};
        for(int j=0;j<9;j++){
            int new_row=row+a[2*j];
            int new_col=col+a[2*j+1];
            if(new_row>=0&&new_row<hidden_row&&new_col>=0&&new_col<hidden_col){
                int index=new_row+new_col*hidden_row+index_picture*hidden_row*hidden_col;
                delta_ph[i*size_pooling+j]=h_sort[index]/p[i];
            }
        }
    }
}

void HPC::compute_delta_vp(){
    int size_v=hidden_row*hidden_col*hidden_channel;
    int size_lcn=lcn_row*lcn_col*hidden_channel;
    int size_delta_vp=size_v*size_lcn;
    memset(delta_vp,0,sizeof(float)*size_delta_vp);
    for(int i=0;i<size_v;i++){
        int index_picture=i/(hidden_row*hidden_col);
        int index_xy=i%(hidden_row*hidden_col);
        int row=index_xy%hidden_row;
        int col=index_xy/hidden_row;
        int a[]={-1,-1,-1,0,-1,1,0,-1,0,0,0,1,1,-1,1,0,1,1};
        for(int j=0;j<9;j++){
            int new_row=row+a[2*j];
            int new_col=col+a[2*j+1];
            if(new_row>=0&&new_row<hidden_row&&new_col>=0&&new_col<hidden_col){
                for(int k=0;k<hidden_channel;k++){
                    //int index=new_row+new_col*hidden_row+k*hidden_row*hidden_col;
                    if(k==index_picture&&j==4) delta_vp[i*size_lcn+j+k*9]=1-lcn_GW[j]; 
                    else delta_vp[i*size_lcn+j+k*9]=-lcn_GW[j];
                }
            }
        }
    }
}

void HPC::compute_delta_lv(){
    int size_v=hidden_row*hidden_col*hidden_channel;
    int size_sigle_v=size_v/hidden_channel;
    int size_lcn=lcn_row*lcn_col*hidden_channel;
    int size_delta_lv=size_v*size_lcn;
    memset(delta_lv,0,sizeof(float)*size_delta_lv);
    for(int i=0;i<size_v;i++){
        int index_picture=i/(hidden_row*hidden_col);
        int index_xy=i%(hidden_row*hidden_col);
        int row=index_xy%hidden_row;
        int col=index_xy/hidden_row;
        int a[]={-1,-1,-1,0,-1,1,0,-1,0,0,0,1,1,-1,1,0,1,1};
        if(c>ln2[i%size_sigle_v]) delta_lv[i*size_lcn+4+index_picture*9]=1/c;
        
        else{
            for(int j=0;j<9;j++){
                int new_row=row+a[2*j];
                int new_col=col+a[2*j+1];
                if(new_row>=0&&new_row<hidden_row&&new_col>=0&&new_col<hidden_col){
                    for(int k=0;k<hidden_channel;k++){
                        int index_v=new_row+new_col*hidden_row+k*hidden_row*hidden_col;
                        int index_ln=i%size_sigle_v;
                        int index_delta_lv=i*size_lcn+k*9+j;
                        if(i==index_v) delta_lv[index_delta_lv]=(ln2[index_ln]*ln2[index_ln]-lc[i]*lcn_GW[j]*lc[index_v])/(ln2[index_ln]*ln2[index_ln]*ln2[index_ln]); 
                        else delta_lv[index_delta_lv]=-lc[i]*lcn_GW[j]*lc[index_v]/(ln2[index_ln]*ln2[index_ln]*ln2[index_ln]);
                    }
                }
            }
        }
    }
}

void HPC::compute_delta_px(){
    int size_p=hidden_row*hidden_col*hidden_channel;
    int M=1;
    int K=pooling_row*pooling_col;
    int N=input_row*input_col*input_channel;
    for(int i=0;i<size_p;i++){
        memset(delta_hx_block,0,sizeof(float)*K*N);
        get_delta_hx_block(i);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1,&delta_ph[i*K],K,delta_hx_block,N,0,&delta_px[i*N],N);
    }
}
void HPC::compute_delta_vx(){
    int size_v=hidden_row*hidden_col*hidden_channel;
    int M=1;
    int K=lcn_row*lcn_col*lcn_channel;
    int N=input_row*input_col*input_channel;
    for(int i=0;i<size_v;i++){
        memset(delta_px_block,0,sizeof(float)*K*N);
        get_delta_px_block(i);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1,&delta_vp[i*K],K,delta_px_block,N,0,&delta_vx[i*N],N);
    }
}
void HPC::compute_delta_lx(){
    int size_v=hidden_row*hidden_col*hidden_channel;
    int M=1;
    int K=lcn_row*lcn_col*lcn_channel;
    int N=input_row*input_col*input_channel;
    for(int i=0;i<size_v;i++){
        memset(delta_vx_block,0,sizeof(float)*K*N);
        get_delta_vx_block(i);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1,&delta_lv[i*K],K,delta_vx_block,N,0,&delta_lx[i*N],N);
    }
}
void HPC::get_delta_hx_block(int i){
    int size_pixelD=input_row*input_col*input_channel;
    int index_picture=i/(hidden_row*hidden_col);
    int index_xy=i%(hidden_row*hidden_col);
    int row=index_xy%hidden_row;
    int col=index_xy/hidden_row;
    int a[]={-1,-1,-1,0,-1,1,0,-1,0,0,0,1,1,-1,1,0,1,1};
    for(int j=0;j<9;j++){
        int new_row=row+a[2*j];
        int new_col=col+a[2*j+1];
        if(new_row>=0&&new_row<hidden_row&&new_col>=0&&new_col<hidden_col){
            int hx_row=new_row+new_col*hidden_row+index_picture*hidden_row*hidden_col;
            for(int k=0;k<size_pixelD;k++){
                delta_hx_block[j*size_pixelD+k]=delta_hx[hx_row*size_pixelD+k];
            }
        }
    }
}

void HPC::get_delta_px_block(int i){
    int size_pixelD=input_row*input_col*input_channel;
    //int index_picture=i/(hidden_row*hidden_col);
    int index_xy=i%(hidden_row*hidden_col);
    int row=index_xy%hidden_row;
    int col=index_xy/hidden_row;
    int a[]={-1,-1,-1,0,-1,1,0,-1,0,0,0,1,1,-1,1,0,1,1};
    for(int j=0;j<9;j++){
        int new_row=row+a[2*j];
        int new_col=col+a[2*j+1];
        if(new_row>=0&&new_row<hidden_row&&new_col>=0&&new_col<hidden_col){
            for(int k=0;k<hidden_channel;k++){
                int px_row=new_row+new_col*hidden_row+k*hidden_row*hidden_col;
                for(int l=0;l<size_pixelD;l++){
                    delta_px_block[(9*k+j)*size_pixelD+l]=delta_px[px_row*size_pixelD+l];
                }
            }
        }
    }
}

void HPC::get_delta_vx_block(int i){
    int size_pixelD=input_row*input_col*input_channel;
    //int index_picture=i/(hidden_row*hidden_col);
    int index_xy=i%(hidden_row*hidden_col);
    int row=index_xy%hidden_row;
    int col=index_xy/hidden_row;
    int a[]={-1,-1,-1,0,-1,1,0,-1,0,0,0,1,1,-1,1,0,1,1};
    for(int j=0;j<9;j++){
        int new_row=row+a[2*j];
        int new_col=col+a[2*j+1];
        if(new_row>=0&&new_row<hidden_row&&new_col>=0&&new_col<hidden_col){
            for(int k=0;k<hidden_channel;k++){
                int px_row=new_row+new_col*hidden_row+k*hidden_row*hidden_col;
                for(int l=0;l<size_pixelD;l++){
                    delta_vx_block[(9*k+j)*size_pixelD+l]=delta_vx[px_row*size_pixelD+l];
                }
            }
        }
    }
}

void HPC::copy_rpixelD(){
    int size_rpixelD=input_row*input_col*input_channel*minibatch;
    for(int i=0;i<size_rpixelD;i++){
        rpixelD[i]=rpixelD_gather[i];
    }
}
void HPC::write_h(){
    int size_h_gather=hidden_row*hidden_col*hidden_channel*minibatch;
    ofstream outfile(h_filename.c_str(),ios::out|ios::binary|ios::app);
    outfile.write((char*)h_sort,sizeof(float)*size_h_gather);
    outfile.close();
}
void HPC::write_p(){
    int size_p_gather=hidden_row*hidden_col*hidden_channel*minibatch;
    ofstream outfile(p_filename.c_str(),ios::out|ios::binary|ios::app);
    outfile.write((char*)p,sizeof(float)*size_p_gather);
    outfile.close();
}

void HPC::write_lcn(){
    int size_lcn_gather=hidden_row*hidden_col*hidden_channel*minibatch;
    ofstream outfile(lcn_filename.c_str(),ios::out|ios::binary|ios::app);
    outfile.write((char*)lcn,sizeof(float)*size_lcn_gather);
    outfile.close();
}

void HPC::zeros_rpixelD(){
    int size_rpixelD=minibatch*input_row*input_col*input_channel;
    for(int i=0;i<size_rpixelD;i++){
        rpixelD[i]=0;
    }
}

void HPC::zeros_rpixelD_gather(){
    int size_rpixelD_gather=minibatch*input_row*input_col*input_channel;
    for(int i=0;i<size_rpixelD_gather;i++){
        rpixelD_gather[i]=0;
    }

}

void HPC::zeros_delta_w(){
    int size_w=hiddenBlocksNumberOfPerProcess*hidden_block_row*hidden_block_col*hidden_channel*filter_row*filter_col*input_channel;
    for(int i=0;i<size_w;i++){
        delta_w[i]=0;
    }
}

void HPC::zeros_w_gather(){
    int size_w_gather=hiddenBlocksNumberOfPerProcess*process_number*hidden_block_row*hidden_block_col*hidden_channel*filter_row*filter_col*input_channel;
    for(int i=0;i<size_w_gather;i++){
        w_gather[i]=0;
    }
}

HPC& HPC::set_minibatch(int mminibatch){
    minibatch=mminibatch;
    return *this;
}

HPC& HPC::set_alpha(float aalpha){
    alpha=aalpha;
    return *this;
}

HPC& HPC::set_ela(float eela){
    ela=eela;
    return *this;
}
HPC& HPC::set_theata(float ttheata){
    theata=ttheata;
    return *this;
}
HPC& HPC::set_c(float cc){
    c=cc;
    return *this;
}
HPC& HPC::set_learning_rate(float llearning_rate){
    learning_rate=llearning_rate;
    return *this;
}
HPC& HPC::set_lcn_channel(int llcn_channel){
    lcn_channel=llcn_channel;
    return *this;
}
HPC& HPC::set_lcn_row(int llcn_row){
    lcn_row=llcn_row;
    return *this;
}
HPC& HPC::set_lcn_col(int llcn_col){
    lcn_col=llcn_col;
    return *this;
}
HPC& HPC::set_inputFileName(string iinputFileName){
    inputFileName=iinputFileName;
    return *this;
}
HPC& HPC::set_h_filename(string hh_filename){
    h_filename=hh_filename;
    return *this;
}
HPC& HPC::set_p_filename(string pp_filename){
    p_filename=pp_filename;
    return *this;
}
HPC& HPC::set_lcn_filename(string llcn_filename){
    lcn_filename=llcn_filename;
    return *this;
}
HPC& HPC::set_picture_number(int ppicture_number){
    picture_number=ppicture_number;
    return *this;
}
float* HPC::get_pixelD() const {
    return pixelD;
}
float* HPC::get_rpixelD() const{
    return rpixelD;
}
float* HPC::get_pixelD_block() const{
    return pixelD_block;
}

float* HPC::get_rpixelD_block() const{
    return rpixelD_block;
}

float* HPC::get_rpixelD_gather() const{
    return rpixelD_gather;
}

float* HPC::get_h() const{
    return h;
}

float* HPC::get_h_gather() const{
    return h_gather;
}
float* HPC::get_w() const{
    return w;
}
float* HPC::get_n_w() const{
    return n_w;
}
float* HPC::get_w_gather() const{
    return w_gather;
}
float* HPC::get_delta_w() const{
    return delta_w;
}
float* HPC::get_delta_w_m() const{
    return delta_w_m;
}
float* HPC::get_h_sort() const{
    return h_sort;
}
float* HPC::get_p() const{
    return p;
}
float* HPC::get_lcn() const{
    return lcn;
}
float* HPC::get_h_delta_alpha() const{
    return h_delta_alpha;
}
float* HPC::get_h_delta_w() const{
    return h_delta_w;
}
float* HPC::get_delta_hx() const{
    return delta_hx;
}
float* HPC::get_delta_px() const{
    return delta_px;
}
float* HPC::get_delta_vx() const{
    return delta_vx;
}
float* HPC::get_delta_lx() const{
    return delta_lx;
}
float* HPC::get_lc() const{
    return lc;
}
float* HPC::get_max_x() const{
    return max_x;
}
int HPC::get_minibatch() const{
    return minibatch;
}
float HPC::get_alpha() const{
    return alpha;
}

