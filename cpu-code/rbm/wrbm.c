#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#define LENGTH 1
#define SIZE2 50000000
#define SIZE3 60000

#define test_length 10000
#define train_length 50000
#define validation_length 10000

#define batch_size 20
#define NUMIN 784
#define NUMHID 500

#define cd_k 15
#define lr 0.1

double test_image[test_length][NUMIN];
double train_image[train_length][NUMIN];
double validation_image[validation_length][NUMIN];

unsigned char buf[SIZE2]={0};

double weight[NUMIN+1][NUMHID];
double vbias[NUMIN]={0};
double grams_w[NUMIN+1][NUMHID];
double grams_v[NUMIN]={0};
double persistent[batch_size][NUMHID];
//double persistent_alone[NUMHID];

int flag_of_persistent[batch_size]={0};
//int flag_of_persistent_alone =0;
double nv_means[NUMIN]={0};
double ph_means[NUMHID]={0};

int ph_sample[NUMHID]={0};
int ph_gibbs_sample[NUMHID]={0};
int nv_gibbs_sample[NUMIN]={0};
int flag_i_flip = 0;

void loaddata_label(char* address,int length);
void loaddata();
void loaddata_image(char* address,int length);
void initalize_w();
double train_rbm(int batch_index);
void updates_parameters();
void get_grams(int index);
void gibbs_hvh_sample(int index,int batch_index);
void sample_h_given_v();
void sample_v_given_h();
double get_reconstruction_cost(int index);
void initalize_w();
double sigmoid(double x);

double get_pseudo_likelihood_cost(int index);
double freeEnergy(int *input);

main(){

	int training_epochs=15;
	int n_chains=20;
	int n_samples=10;
	int n_train_batches;
	int epoch,batch_index;
	
	bzero(persistent,sizeof(persistent));
	bzero(grams_w,sizeof(grams_w));
	bzero(weight,sizeof(weight));
	
	loaddata();
	
	initalize_w();
	
	n_train_batches = train_length/batch_size;
	for(epoch=0;epoch<training_epochs;epoch++){
		double mean_cost=0;
		
		for(batch_index=0;batch_index<n_train_batches;batch_index++){
			mean_cost+=train_rbm(batch_index);
			flag_i_flip=(flag_i_flip+1)%NUMIN;
			printf("Training epoch: %d\n",epoch);
			printf("\tbatch_index: %d \n",batch_index);
			if(batch_index==0)  printf("\t\tmean_cost:%lf\n",mean_cost);
			else printf("\t\tmean_cost:%lf\n",mean_cost/(batch_index+1));
		}
/*		if(epoch==0){
			FILE *fp;
			if((fp==fopen("rbm_weight.txt","w"))==NULL){
				printf("cannot open the file.");
			}
			fwrite(weight[1],sizeof(double),NUMIN*NUMHID,fp);
		}
		//printf("Training epoch %d,cost is %lf\n",epoch,mean_cost/n_train_batches);

*/	}		    
}

double train_rbm(int batch_index){
	double cost=0.0;
	int i,j;
	int index = batch_index*batch_size;
	for(j=0;i<NUMHID;j++){
		for(i=0;i<NUMIN+1;i++){
			grams_w[i][j]=0;		
		}
	}
	for(i=0;i<NUMIN;i++){
		grams_v[i]=0;	
	}
	
	for(i=0;i<batch_size;i++){
		gibbs_hvh_sample(index+i,i);
		get_grams(index+i);
		//cost+=get_reconstruction_cost(index+i);
	}
	updates_parameters();
	for(i=0;i<batch_size;i++){
		cost+=get_pseudo_likelihood_cost(index+i);
	}
	cost = cost/batch_size;
	//printf("train_rbm average of a minibatch cost: %lf\n",cost);
	return cost;
}
double get_pseudo_likelihood_cost(int index){
	int input[NUMIN]={0};
	int input_i_flip[NUMIN]={0};
	int i;
	double cost=0;
	for(i=0;i<NUMIN;i++){
		if(train_image[index][i]>=0.5){
			input[i]=1;
			input_i_flip[i]=1;
		}
	}
	input_i_flip[flag_i_flip]=1-input_i_flip[flag_i_flip];
	cost=-NUMIN*log(1+exp(freeEnergy(input)-freeEnergy(input_i_flip)));
	return cost;
}
double freeEnergy(int *input){
	double vbias_term=0;
	double hidden_term=0;
	double freeEnergy=0;
	double hidden_term_middle=0;
	int i,j;
	for(i=0;i<NUMIN;i++){
		vbias_term+=input[i]*vbias[i];
	}
	for(j=0;j<NUMHID;j++){
		hidden_term_middle=weight[0][j];
		for(i=0;i<NUMIN;i++){
			hidden_term_middle+=input[i]*weight[i+1][j];
		}
		hidden_term +=log(1+exp(hidden_term_middle));
	}
	freeEnergy=-hidden_term-vbias_term;
	return freeEnergy;
}
void updates_parameters(){
	int i,j;
	for(j=0;j<NUMHID;j++){
		weight[0][j]=weight[0][j]+lr*grams_w[0][j]/batch_size;
		for(i=0;i<NUMIN;i++){
			weight[i+1][j]=weight[i+1][j]+lr*grams_w[i+1][j]/batch_size;
		}	
	}
	for(i=0;i<NUMIN;i++){
		vbias[i]=vbias[i]+lr*grams_v[i]/batch_size;	
	}
}
void get_grams(int index){
	int i,j;
	for(j=0;j<NUMHID;j++){
		grams_w[0][j]+=ph_sample[j]-ph_means[j];
		for(i=0;i<NUMIN;i++){
			grams_w[i+1][j]+=ph_sample[j]*train_image[index][i]-ph_means[j]*nv_gibbs_sample[i];	
		}	
	}
	for(i=0;i<NUMIN;i++){
		grams_v[i]+=train_image[index][i]-nv_gibbs_sample[i];
	}
}
void gibbs_hvh_sample(int index,int batch_index){
	int i,j,k;
	for(i=0;i<NUMIN;i++){
		nv_gibbs_sample[i]=train_image[index][i];	
	}
	sample_h_given_v();
	for(j=0;j<NUMHID;j++){
		ph_sample[j]=ph_gibbs_sample[j];	
	}
//	if(flag_of_persistent[batch_index]==1){
		for(j=0;j<NUMHID;j++){
			ph_gibbs_sample[j] = persistent[batch_index][j];
		}

//	}
	for(k=0;k<cd_k;k++){
		sample_v_given_h();
		sample_h_given_v();
	}
	for(j=0;j<NUMHID;j++){
		persistent[batch_index][j]=ph_gibbs_sample[j];
	}
//	flag_of_persistent[batch_index]=1;
}

void sample_h_given_v(){
	int i,j,k;
	double h_mean=0.0;
	double pre_sigmoid=0.0;
	//time_t rawtime;
	for(j=0;j<NUMHID;j++){
		pre_sigmoid=weight[0][j];
		for(i=0;i<NUMIN;i++){
			pre_sigmoid+=nv_gibbs_sample[i]*weight[i+1][j];	
		}	
		h_mean=1/(1+exp(-pre_sigmoid));
		ph_means[j]=h_mean;
		//time(&rawtime);
		//srand((unsigned)rawtime);
		k=rand()%100;
		if(k/100.0>h_mean) ph_gibbs_sample[j]=0;
		else	ph_gibbs_sample[j]=1;
	}
}
void sample_v_given_h(){
	int i,j,k;
	double v_mean=0.0;
	double pre_sigmoid=0.0;
	//time_t rawtime;
	for(i=0;i<NUMIN;i++){
		pre_sigmoid=vbias[i];
		for(j=0;j<NUMHID;j++){
			pre_sigmoid+=ph_gibbs_sample[j]*weight[i+1][j];	
		}	
		v_mean=1.0/(1.0+exp(-pre_sigmoid));
		nv_means[i]=v_mean;
		//time(&rawtime);
		//srand((unsigned)rawtime);
		k=rand()%100;
		if(k/100.0>v_mean) nv_gibbs_sample[i]=0;
		else	nv_gibbs_sample[i]=1;
	}
}

double get_reconstruction_cost(int index){
	int i;
	double cost=0;
	for(i=0;i<NUMIN;i++){
		cost+=train_image[index][i]*log(nv_means[i])+(1-train_image[index][i])*log(1-nv_means[i]);
	}
	return cost;
}

void initalize_w(){
	//bzero(WeightHO, sizeof(WeightHO));
	int i,j,k;
	double low = -4*sqrt(6.0/(NUMIN+NUMHID));
	double high = 4*sqrt(6.0/(NUMIN+NUMHID));
	//double low = -sqrt(6.0/(NUMIN+NUMHID));
	//double high = sqrt(6.0/(NUMIN+NUMHID));
	for(j=0;j<NUMHID;j++){
		weight[0][j]=0.0;	
	}
	for(i=0;i<NUMIN;i++){
		for(j=0;j<NUMHID;j++){
			k=rand()%200;
			if(k<100){
				weight[i+1][j]=(k/100.0)*low;		
			}
			else{
				weight[i+1][j]=(k-100)/100.0*high;			
			}
		}
	}
	
	
}


void loaddata(){
	
    char *address[]={"../mlp/t10k-labels.idx1-ubyte","../mlp/train-labels.idx1-ubyte","../mlp/t10k-images.idx3-ubyte","../mlp/train-images.idx3-ubyte"};
    
    loaddata_image(address[2],0);
    loaddata_image(address[3],1);
    
}

void loaddata_image(char* address,int length){
    FILE *fp;
    int i,j;
    int index=16;
    fp = fopen(address,"r");
    if(fp!=NULL&&fread(buf,1,SIZE2/LENGTH,fp) >=0){
        puts("success");
        if(length==0){
                for(i=0;i<test_length;i++){
                    for(j=0;j<784;j++){
                        test_image[i][j]=(int)buf[index++]/255.0;
                    }
                }
        }
        else{
            for(i=0;i<SIZE3;i++){
                if(i<train_length){
                    for(j=0;j<784;j++){
                        train_image[i][j]=(int)buf[index++]/255.0;
                    }
                }
                else{
                    for(j=0;j<784;j++){
                        validation_image[i-train_length][j]=(int)buf[index++]/255.0;
                    }
               }
           }
       }
        
    }
    else
        perror("open the file");
    fclose(fp);
}

