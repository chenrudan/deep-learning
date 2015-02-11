#include<iostream>
#include<fstream>
using namespace std;

void lcn(float* p,float* lcn,int size);
int main(){
    string inputfile="face1_p_t9.dat";
    int inputnumber=5000;
    int size=88*88*3;
    string outputfile="face1_lc_t9.dat";

    ifstream fp(inputfile.c_str(),ios::in|ios::binary);
    ofstream fp1(outputfile.c_str(),ios::out|ios::binary);

    float* p=new float[size];
    float* lc=new float[size];
    for(int i=0;i<inputnumber;i++){
        fp.read((char*)p,sizeof(float)*size);
        lcn(p,lc,size);
        fp1.write((char*)lc,sizeof(float)*size);
    }

    delete[] p;
    delete[] lc;
    fp.close();
    fp1.close();
    return 0;
}

void lcn(float* p,float* lcn,int size){
    int hidden_channel=3;
    int hidden_row=88;
    int hidden_col=88;
    int size_lcn=hidden_row*hidden_col;

    float* lcn_GW=new float[9];
    float* lcn_1=new float[size_lcn];
    lcn_GW[0]=0.0625/hidden_channel;
    lcn_GW[1]=0.125/hidden_channel;
    lcn_GW[2]=0.0625/hidden_channel;
    lcn_GW[3]=0.125/hidden_channel;
    lcn_GW[4]=0.25/hidden_channel;
    lcn_GW[5]=0.125/hidden_channel;
    lcn_GW[6]=0.0625/hidden_channel;
    lcn_GW[7]=0.125/hidden_channel;
    lcn_GW[8]=0.0625/hidden_channel;
    int a[]={-1,-1,-1,0,-1,1,0,-1,0,0,0,1,1,-1,1,0,1,1};
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
                    int index_p=index+k*size_lcn;
                    sum+=p[index_p]*lcn_GW[j];
                }    
            }    
        }    
        lcn_1[i]=sum;
    }
    for(int i=0;i<size;i++){
        lcn[i]=p[i]-lcn_1[i%size_lcn];
    }
    delete[] lcn_GW;
    delete[] lcn_1;
}
