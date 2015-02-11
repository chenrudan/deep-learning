#include<iostream>
#include<fstream>
#include<sstream>
#include"mpi.h"
extern "C"{
#include"cblas.h"
#include<time.h>
#include<stdlib.h>
#include<math.h>
#include<unistd.h>
}
#include"HPC.h"

using namespace std;

int main(int argc,char **argv){
    int rank;
    int process_number;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&process_number);
    MPI_Get_processor_name(processor_name,&namelen);

    //输入相关
    const int input_row=96;
    const int input_col=96;
    const int input_channel=1;
    const string inputFileName="./testface/lfwfaces_valid.bin";
    //const int picture_number=5000;
    const int picture_number=210;

    //模型相关，filter
    const int filter_row=10;
    const int filter_col=10;
    const int step_size=2;
    //hidden layer
    const int hidden_block_row=2;
    const int hidden_block_col=2;
    const int hidden_channel=3;
    const int hidden_row=((input_row-filter_row)/step_size+1)*hidden_block_row;
    const int hidden_col=((input_col-filter_col)/step_size+1)*hidden_block_col;

    //pooling layer
    const int pooling_row=3;
    const int pooling_col=3;


    HPC firstlayer(rank,process_number,processor_name,input_row,input_col,input_channel,inputFileName,picture_number,filter_row,filter_col,step_size,hidden_block_row,hidden_block_col,hidden_channel,pooling_row,pooling_col);

    //trainning parameters
    //const int minibatch=40;
    //const int n_epochs= 1000;
    //test parameters
    const int minibatch=10;
    const int n_epochs= 1;
    //alpha*W*X=h
    //ela 0.01 alpha 0.12  learning_rate 20 alpha_learning_rate=0.001 u 0.005
    float alpha=0.175592;
    const float theata=0.0001;
    //E=E1+ela*E2
    const float ela= 0.003;
    //w=w+learning_rate*delta_W+u*delta_W_mconst float learning_rate=12;
    const float learning_rate= 100;
    const float alpha_learning_rate=0.001;
    
    const float u=0.005;
    const string outfilename="face1_w_t1.dat";
    const string h_filename="./testface/lfwface1_h_valid.dat";
    const string p_filename="./testface/lfwface1_p_valid.dat";
    const string lcn_filename="./testface/lfwface1_outcome_valid.dat";
    const string R_0_file="face1_R0_t1.dat";
    const string R_n_file="face1_Rn_t1.dat";
    if(rank==0){
        cout<<"************************************************"<<endl;
        cout<<"-----------train_face1--------"<<endl;
        cout<<"input:"<<inputFileName<<endl;
        cout<<"input_number:"<<picture_number<<endl;
        cout<<"n_epoch:"<<n_epochs<<endl;
        cout<<"alpha:"<<alpha<<endl;
        cout<<"ela:"<<ela<<endl;
        cout<<"learning_rate:"<<learning_rate<<endl;
        cout<<"u:"<<u<<endl;
        cout<<"alpha_learning_rate:"<<alpha_learning_rate<<endl;
        cout<<"outfilename:"<<outfilename<<endl;
        cout<<"************************************************"<<endl;
    }


    //lcn layer
    const int lcn_channel=3;
    const int lcn_row=3;
    const int lcn_col=3;

    firstlayer.set_minibatch(minibatch);
    firstlayer.set_alpha(alpha);
    firstlayer.set_theata(theata);
    firstlayer.set_ela(ela);
    firstlayer.set_learning_rate(learning_rate);
    firstlayer.set_lcn_channel(lcn_channel);
    firstlayer.set_lcn_row(lcn_row);
    firstlayer.set_lcn_col(lcn_col);
    firstlayer.set_h_filename(h_filename);
    firstlayer.set_p_filename(p_filename);
    firstlayer.set_lcn_filename(lcn_filename);


    int numberOfminibatch=picture_number/minibatch;
    int hiddenBlocksNumberOfPerProcess=(hidden_row/hidden_block_row)*(hidden_col/hidden_block_col)/process_number;
    int size_pixelD=input_row*input_col*input_channel*minibatch;
    int size_w_block=hidden_block_row*hidden_block_col*hidden_channel*filter_row*filter_col*input_channel;
    int size_w=hiddenBlocksNumberOfPerProcess*size_w_block;
    int size_h_block=minibatch*hidden_block_row*hidden_block_col*hidden_channel;
    int size_h=size_h_block*hiddenBlocksNumberOfPerProcess;
    int size_h_gather=minibatch*hidden_row*hidden_col*hidden_channel;

    srand(time(NULL));
    firstlayer.malloc();
    float *w_gather = firstlayer.get_w_gather();
    float *w = firstlayer.get_w();
    //read w
    if(rank == 0){
        firstlayer.loaddata_w("./face1_w/1999face1_w_t9.dat");
    }
    //read w and bcast
    MPI_Scatter(w_gather, size_w, MPI_FLOAT, w, size_w, MPI_FLOAT, 0, MPI_COMM_WORLD);
//    firstlayer.initlize_w();
    firstlayer.initlize_lcn_GW();
    firstlayer.zeros_delta_w();

    time_t time1=time(NULL);
/*
    for(int epoch=0;epoch<n_epochs;epoch++){

        for(int minibatch_index=0;minibatch_index<numberOfminibatch;minibatch_index++){
            firstlayer.set_inputFileName(inputFileName);
            firstlayer.set_picture_number(picture_number);
            if(rank==0){
                //firstlayer.loaddata_minist(minibatch_index);
                firstlayer.loaddata_Float(minibatch_index);
                //need whitening
                firstlayer.preprocess();
            }
            //0号进程将预处理后的输入图片数据广播给其他进程，同样存在pixelD中
            float *pixelD=firstlayer.get_pixelD();
            //float *w=firstlayer.get_w();
            float *delta_w=firstlayer.get_delta_w();
            //float *pixelD_block=firstlayer.get_pixelD_block();
            float *h=firstlayer.get_h();
            float *h_gather=firstlayer.get_h_gather();
            //float *rpixelD_block=firstlayer.get_rpixelD_block();
            
            MPI_Bcast(pixelD,size_pixelD,MPI_FLOAT,0,MPI_COMM_WORLD);
            firstlayer.normalize_w();
            //copy delta_w to delta_w_m
            //initlize delta_w momentum
            firstlayer.initlize_delta_w_m();
            firstlayer.zeros_rpixelD();
            firstlayer.zeros_rpixelD_gather();
            firstlayer.zeros_delta_w();
            //每个进程循环训练所分配的blocks
            firstlayer.compute_rpixelD();
            //搜集叠加后的rpixelD，存入rpixelD_gather中
            float* rpixelD=firstlayer.get_rpixelD();
            float* rpixelD_gather=firstlayer.get_rpixelD_gather();
            MPI_Allreduce(rpixelD,rpixelD_gather,size_pixelD,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
            firstlayer.copy_rpixelD();

            if(rank==0){
                cout<<"-----------pixelD---------rpixelD------"<<endl;
                for(int i=200;i<300;i++){
                    cout<<pixelD[i]<<":"<<rpixelD_gather[i]<<" ";
                }
                cout<<endl;
            }

            //just store 5 minibatch pictures
            if(epoch==0&&rank==0&&minibatch_index<20){
                ofstream out(R_0_file.c_str(),ios::out|ios::binary|ios::app);
                out.write((char*)rpixelD_gather,size_pixelD*sizeof(float));
                out.close();
            }
            if(epoch==n_epochs-1&&rank==0&&minibatch_index<20){
                ofstream out(R_n_file.c_str(),ios::out|ios::binary|ios::app);
                out.write((char*)rpixelD_gather,size_pixelD*sizeof(float));
                out.close();
            }
            catlas_saxpby(size_pixelD,-1,pixelD,1,1,rpixelD_gather,1);
            //gather h for pooling
            MPI_Allgather(h,size_h,MPI_FLOAT,h_gather,size_h,MPI_FLOAT,MPI_COMM_WORLD);

            //pooling
            if(rank==1){
                firstlayer.sort_h_gather();
                firstlayer.pooling();
                firstlayer.compute_h_delta_alpha();
                firstlayer.compute_h_delta_w();
            }
            float* h_sort=firstlayer.get_h_sort();
            float* h_delta_w=firstlayer.get_h_delta_w();
            float* h_delta_alpha=firstlayer.get_h_delta_alpha();
            float* p=firstlayer.get_p();

            MPI_Bcast(h_sort,size_h_gather,MPI_FLOAT,1,MPI_COMM_WORLD);
            MPI_Bcast(h_delta_w,size_h_gather,MPI_FLOAT,1,MPI_COMM_WORLD);
            MPI_Bcast(h_delta_alpha,size_h_gather,MPI_FLOAT,1,MPI_COMM_WORLD);
            MPI_Bcast(p,size_h_gather,MPI_FLOAT,1,MPI_COMM_WORLD);
            //show h
            if(rank==0){
                cout<<"-----------h_sort--------"<<endl;
                for(int i=0;i<300;i++){
                    cout<<h_sort[i]<<" ";
                }
                cout<<endl;
                cout<<"-----------p------------"<<endl;
                for(int i=0;i<300;i++){
                    cout<<p[i]<<" ";
                }
                cout<<endl;
            }

            //comput the minimize outcome ,R-x and pooling
            if(rank==0){
                //comput R-x cost
                float R_cost=0;
                for(int i=0;i<size_pixelD;i++){
                    R_cost+=rpixelD_gather[i]*rpixelD_gather[i];
                }
                //comput p cost
                float P_cost=0;
                for(int i=0;i<size_h_gather;i++){
                    P_cost+=p[i];
                }
                P_cost=ela*P_cost;
                cout<<"-------------------cost---------------------------"<<endl;
                cout<<"epoch:"<<epoch<<"\tminibatch_inde:"<<minibatch_index<<"/"<<numberOfminibatch<<endl;
                cout<<"\tR_cost:"<<R_cost<<"\tP_cost:"<<P_cost<<"\tall_cost:"<<R_cost+P_cost<<endl;
            }
            //compute delta_alpha
            float delta_alpha=0;
            if(rank==0){
                float alpha2=firstlayer.compute_delta_alpha2();
                float alpha1=firstlayer.compute_delta_alpha1();
                delta_alpha=alpha1+alpha2;
                cout<<"-----------------delta_alpha-----------"<<endl;
                cout<<"alpha:"<<alpha<<"\tdelta_alpha1:"<<alpha1<<"\tdelta_alpha2:"<<alpha2<<"\tdelta_alpha:"<<alpha_learning_rate*delta_alpha<<endl;
            }
            MPI_Bcast(&delta_alpha,1,MPI_FLOAT,0,MPI_COMM_WORLD);
            alpha=alpha-alpha_learning_rate*delta_alpha;
            firstlayer.set_alpha(alpha);

            //compute delta_w
            firstlayer.compute_delta_w();

            //更新W=-learning_rate*delta_w+W
            float* delta_w_m=firstlayer.get_delta_w_m();

            float w_absolute_changes=0;
            float w_absolute_changes_all=0;
            float w_relative_changes=0;
            float w_relative_changes_all=0;
            for(int w_index=0;w_index<size_w;w_index++){
                w_absolute_changes+=fabs(delta_w[w_index]);
                w_relative_changes+=fabs(delta_w[w_index]+delta_w_m[w_index]);
            }   
            MPI_Allreduce(&w_absolute_changes,&w_absolute_changes_all,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
            MPI_Allreduce(&w_relative_changes,&w_relative_changes_all,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
            if(rank==0){
                cout<<"------------------w_changes-----------------------------------"<<endl;
                cout<<"w_absolute_changes:"<<w_absolute_changes<<"\tw_absolute_changes_all:"<<w_absolute_changes_all<<endl;
                cout<<"w_relative_changes:"<<w_relative_changes<<"\tw_relative_changes_all:"<<w_relative_changes_all<<endl;
                cout<<"------------w-----------------------"<<endl;
                for(int i=0;i<100;i++){
                    cout<<w[i]<<":"<<learning_rate*delta_w[i]<<"\t";
                }   
                cout<<endl;
            }

            catlas_saxpby(size_w,-1*learning_rate,delta_w,1,1,w,1);
            catlas_saxpby(size_w,-1*u,delta_w_m,1,1,w,1);
            //sleep(5);
        }
        if((epoch+1)%100==0){
            firstlayer.zeros_w_gather();
            //float *w=firstlayer.get_w();
            //float *w_gather=firstlayer.get_w_gather();
            MPI_Gather(w,size_w,MPI_FLOAT,w_gather,size_w,MPI_FLOAT,0,MPI_COMM_WORLD);
            int size_w_gather=size_w*process_number;
            int index=epoch;
            //int index=epoch;
            stringstream stream;
            stream<<index+2000;
            string w_name=stream.str()+outfilename;
            if(rank==0){
                ofstream outfile(w_name.c_str(),ios::out|ios::binary);
                outfile.write((char*)w_gather,sizeof(float)*size_w_gather);
                outfile.close();
            }


        }
    }
    time_t time2=time(NULL);
    //firstlayer.normalize_w();
//    float *w=firstlayer.get_w();
//    *w_gather=firstlayer.get_w_gather();
    MPI_Gather(w,size_w,MPI_FLOAT,w_gather,size_w,MPI_FLOAT,0,MPI_COMM_WORLD);
    int size_w_gather=size_w*process_number;
    if(rank==0){
        cout<<"---w_gather--------:"<<size_w_gather<<endl;
        for(int i=0;i<10;i++){
            cout<<w_gather[i]<<" ";
        }
        cout<<endl;
        for(int i=size_w_gather-10;i<size_w_gather;i++){
            cout<<w_gather[i]<<" ";
        }
        cout<<endl;
        ofstream outfile(outfilename.c_str(),ios::out|ios::binary);
        outfile.write((char*)w_gather,sizeof(float)*size_w_gather);
        outfile.close();
        cout<<"time cost:"<<(time2-time1)/n_epochs<<" s per epoch"<<endl;
    }
    //output to the next layer
    */
    firstlayer.set_inputFileName(inputFileName);
    firstlayer.set_picture_number(picture_number);
    for(int minibatch_index=0;minibatch_index<numberOfminibatch;minibatch_index++){
        if(rank==0){
            //firstlayer.loaddata_minist(minibatch_index);
            firstlayer.loaddata_Float(minibatch_index);
            firstlayer.preprocess();
        }
        //0号进程将预处理后的输入图片数据广播给其他进程，同样存在pixelD中
        float *pixelD=firstlayer.get_pixelD();
        float *n_w=firstlayer.get_n_w();
        float *pixelD_block=firstlayer.get_pixelD_block();
        float *h=firstlayer.get_h();
        float *h_gather=firstlayer.get_h_gather();

        MPI_Bcast(pixelD,size_pixelD,MPI_FLOAT,0,MPI_COMM_WORLD);
        firstlayer.normalize_w();
        //why wrong 
        //firstlayer.compute_rpixelD();
        for(int i=0;i<hiddenBlocksNumberOfPerProcess;i++){
            firstlayer.initlize_pixelD_block(i); 
            int M=hidden_block_row*hidden_block_col*hidden_channel;
            int K=filter_row*filter_col*input_channel;
            int N=minibatch;
            //计算W[M*K]* X[K*N]= H[M*N]
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,alpha,&n_w[i*size_w_block],K,pixelD_block,N,0,&h[i*size_h_block],N);  
        }
        MPI_Allgather(h,size_h,MPI_FLOAT,h_gather,size_h,MPI_FLOAT,MPI_COMM_WORLD);
        //pooling
        if(rank==0){
            firstlayer.sort_h_gather();
            firstlayer.pooling();
            firstlayer.Lcn();
            firstlayer.write_h();
            firstlayer.write_p();
            firstlayer.write_lcn();
            float* lcn=firstlayer.get_lcn();
            float* p=firstlayer.get_p();
            float* h_sort=firstlayer.get_h_sort();
            cout<<"--------------------h_sort--------------------"<<endl;
            for(int i=0;i<200;i++){
                cout<<h_sort[i]<<" ";
            }
            cout<<endl;

            cout<<"--------------------p--------------------"<<endl;
            for(int i=0;i<200;i++){
                cout<<p[i]<<" ";
            }
            cout<<endl;
            cout<<"-----------------lcn---------------------"<<endl;
            for(int i=0;i<200;i++){
                cout<<lcn[i]<<" ";
            }
            cout<<endl;

        }
    }
    firstlayer.free();
    MPI_Finalize();
    return 0;
}
