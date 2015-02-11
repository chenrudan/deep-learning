#include <stdio.h>
#include <time.h> 
#include <math.h>
#include <stdlib.h>

#define TRAIN_LENGTH 50000
#define TEST_LENGTH 10000

//输入节点有500个，隐藏层节点有500个，输出节点有10个
#define NUMVIS 784
#define NUMHID 500

#define BATCH_SIZE 20

#define bool    _Bool  
#define true    1  
#define false   0

double Train_Pixel[TRAIN_LENGTH][NUMVIS];
double Test_Pixel[TEST_LENGTH][NUMVIS];
double Weight[NUMVIS][NUMHID];
double hbias[NUMHID];
double vbias[NUMVIS];

//中间变量
double pre_sigmoid_h1[BATCH_SIZE][NUMHID], h1_mean[BATCH_SIZE][NUMHID], h1_sample[BATCH_SIZE][NUMHID], h0_sample[BATCH_SIZE][NUMHID];
double pre_sigmoid_v1[BATCH_SIZE][NUMVIS], v1_mean[BATCH_SIZE][NUMVIS], v1_sample[BATCH_SIZE][NUMVIS], v0_sample[BATCH_SIZE][NUMVIS];

double Input[BATCH_SIZE][NUMVIS];
double persistent[BATCH_SIZE][NUMHID];
double vis_persistent[BATCH_SIZE][NUMVIS];
double update_W[BATCH_SIZE][NUMHID];


int bit_i_idx;


void initialize_W();
void load_data();
void sample_h_given_v();
void sample_v_given_h();
double get_pseudo_likelihood_cost();
double get_reconstruction_cost();
void updates_parameter();
double rbm(bool);
void test_vhv();
double free_energy();

void sample_h_given_v(){
	
	int i,j,k,rand_h;
	double m;		
//	for(j=0 ; j<NUMHID ; j++){
//		rand_h[j] = rand()/(RAND_MAX/100 + 1);		
//	}	
	for(i=0; i<BATCH_SIZE; i++){
		for(j=0 ; j<NUMHID ; j++){
			m=0;
			for(k=0 ; k<NUMVIS ; k++){
				m += v0_sample[i][k]*Weight[k][j];
			}			
			pre_sigmoid_h1[i][j] = m + hbias[j];
			h1_mean[i][j] = 1.0/(1.0 + exp(-pre_sigmoid_h1[i][j])); 
			rand_h = rand()%100;
			if( rand_h/100.0<h1_mean[i][j] ){
				h1_sample[i][j] = 1.0;
			}else{
				h1_sample[i][j] = 0.0;
			}			
		}
	}	
}
void sample_v_given_h(){
	
	int i,j,k,rand_v;
	double m;	
//	for(j=0 ; j<NUMVIS ; j++){
//		rand_v[j] = rand()/(RAND_MAX/100 +1);		
//	}	
	for(i=0 ; i<BATCH_SIZE; i++){
		for(j=0 ; j<NUMVIS ; j++){
			m=0;
			for(k=0 ; k<NUMHID ; k++){
				m += h0_sample[i][k]*Weight[j][k];
			}
			pre_sigmoid_v1[i][j] = m + vbias[j];
			v1_mean[i][j] = 1.0/(1.0 + exp(-pre_sigmoid_v1[i][j])); 
			rand_v = rand()%100;
			if( rand_v/100.0<v1_mean[i][j] ){
				v1_sample[i][j] = 1.0;
			}else{
				v1_sample[i][j] = 0.0;
			}			
		}
	}	
}

double rbm(bool chain_start){
	int i,j,k,w;
	double cost;
	for(i=0 ; i<BATCH_SIZE ; i++){
		for(j=0 ; j<NUMVIS ;j++){		
			v0_sample[i][j] = Input[i][j];			
		}
	}
	//得到ph_sample,即sample_hv里面的h1_sample,这个值是算出来的
	sample_h_given_v();
	//确定起始位置为ph_sample or PCD 的起始persistent	
	//hvh过程，由h1_sample先得到h0_sample,再得到v1_sample
	
	for(k=0 ; k<15 ; k++){		
		for(i=0 ; i<BATCH_SIZE ; i++){
			for(j=0 ; j<NUMHID ;j++){
				if(k==0){
				//确定起始位置为ph_sample or PCD 的起始persistent
					if(chain_start){		
						h0_sample[i][j] = h1_sample[i][j];
					}
					else{
						h0_sample[i][j] = persistent[i][j];
					}					
				}else{		
					h0_sample[i][j] = h1_sample[i][j];
				}
				update_W[i][j] = h1_sample[i][j];				
			}			
		}
		sample_v_given_h();
		for(i=0 ; i<BATCH_SIZE ; i++){
			for(j=0 ; j<NUMVIS ;j++){		
					v0_sample[i][j] = v1_sample[i][j];					
			}
		}
		sample_h_given_v();		
//		printf("the time is %lf\n", (double)(end_time-start_time)/CLOCKS_PER_SEC);
	}	
	//更新参数
	updates_parameter();
	
	//结果中的v1和h1便是最后的nv,nh,并且将最后的h1赋值给persistent/	if(!chain_start){
	for(i=0 ; i<BATCH_SIZE ; i++){
		for(j=0 ; j<NUMHID ; j++){
			persistent[i][j] = h1_sample[i][j];
		}
	}
	if(!chain_start){
		cost = get_pseudo_likelihood_cost();
	
	}else{
		cost = get_reconstruction_cost();
	}
	return cost;	
}

double get_pseudo_likelihood_cost(){
	int i,j;
	double  diff=0;
	double fexi, fexi_flip;
	//fe(xi)
	for(i=0 ; i<BATCH_SIZE ; i++){
		fexi = free_energy(i);	
		Input[i][bit_i_idx] = 1-Input[i][bit_i_idx];		
		fexi_flip = free_energy(i);	
		diff += NUMVIS*log(1.0/(1.0 +exp( fexi- fexi_flip)));
	}					
	bit_i_idx = (bit_i_idx+1)%NUMVIS;	 
	return diff/BATCH_SIZE;	
}

double free_energy(int index){
	int j,k;
	double vbias_term ,wx_b;
	double hidden_term ;
	vbias_term = 0;	
	for(j=0 ; j<NUMVIS ; j++){
		vbias_term += ((int)(Input[index][j] + 0.5))*vbias[j];
	}	
	hidden_term = 0;
	for(k=0 ; k<NUMHID ; k++){
		wx_b = 0;
		for(j=0 ; j<NUMVIS ;j++){
			wx_b += ((int)(Input[index][j] + 0.5))*Weight[j][k];
		}												  												
		hidden_term += log(1 + exp(wx_b + hbias[k]));		
	}		
	return  (-hidden_term-vbias_term);
}


double get_reconstruction_cost(){
	int i,j;
	double cross_entropy0 ;
	double cross_entropy1 ;
	for(i=0 ; i<BATCH_SIZE ; i++){
		cross_entropy0 = 0;
		cross_entropy1 = 0;
		for(j=0 ; j<NUMVIS ; j++){
			cross_entropy0 += Input[i][j]*log(1.0/(1.0 + exp(-pre_sigmoid_v1[i][j]))); 
			cross_entropy1 += (1-Input[i][j])*log(1-(1.0/(1.0 + exp(-pre_sigmoid_v1[i][j]))));		
		}
	}
	return (cross_entropy0+cross_entropy1)/BATCH_SIZE;	
}
void updates_parameter(){
	double learning_rate = 0.1;	
	int i,j,k;
	double h1x1,qnxn,x1xn,h1qn;
	for(i=0 ; i<NUMVIS ; i++){
		x1xn = 0;
		for(j=0 ; j<NUMHID ;j++){
			h1x1 = 0;
			qnxn = 0;			
			for(k=0; k<BATCH_SIZE ; k++){
				h1x1 += update_W[k][j]*Input[k][i];
				qnxn += h1_mean[k][j]*v1_sample[k][i];
				
			}
			Weight[i][j] = Weight[i][j] + learning_rate*( h1x1/BATCH_SIZE - qnxn/BATCH_SIZE);
		}
		for(k=0; k<BATCH_SIZE ; k++){
			x1xn += Input[k][i] - v1_sample[k][i]; 
		}
		vbias[i] = vbias[i] + learning_rate*(x1xn/BATCH_SIZE);
	}
	for(j=0 ; j<NUMHID ; j++){
		h1qn = 0;
		for(k=0; k<BATCH_SIZE ; k++){
			h1qn += update_W[k][j] - h1_mean[k][j];
		}
		hbias[j] = hbias[j] + learning_rate*(h1qn/BATCH_SIZE);
	}
}

void initialize_W(){
	int i,j,k;
	double low = -4*sqrt(6.0 /(NUMVIS+NUMHID));
	double high = 4*sqrt(6.0/(NUMVIS+NUMHID));
	
	srand(1234);
	for(i=0 ; i<NUMVIS ; i++){
		for(j=0 ; j<NUMHID ; j++){
			k=rand()%200;
			if(k<100){
				Weight[i][j] =(k/100.0)*low;		
			}
			else{
				Weight[i][j] =(k-100)/100.0*high;			
			}
		}
	}
}

void load_data(){
	
	int i,j,k, startPos1 = 12, startPos2 =8 ;
	unsigned char *fa1, *fa3;
	FILE *fp1, *fp3;	
	fa1 = (char *)malloc(((TRAIN_LENGTH)*NUMVIS)*sizeof(char));
	fa3 = (char *)malloc((TEST_LENGTH*NUMVIS)*sizeof(char));
	
	fp1 = fopen("../mlp/train-images.idx3-ubyte", "r");
    	fp3 = fopen("../mlp/t10k-images.idx3-ubyte", "r");	

	fread(fa1 , 1, (TRAIN_LENGTH*NUMVIS) , fp1);
	fread(fa3 , 1, (TEST_LENGTH*NUMVIS) , fp3);

        //初始化训练集元素
	for(i=0; i<TRAIN_LENGTH ; i++){
		for(j=0; j<NUMVIS; j++){ 
		//bring the input image value into 0~1
			Train_Pixel[i][j] = ((int)fa1[startPos1])/255.0;
			startPos1++;					
		}				
	}	
	free(fa1);
	//初始化测试集
	startPos1 = 12;
	startPos2 = 8; 
	for(i=0; i<TEST_LENGTH ; i++){
		for(j=0; j<NUMVIS; j++){ 
		//bring the input image value into 0~1
			Test_Pixel[i][j] = ((int)fa3[startPos1])/255.0;
			startPos1++;					
		}			
	}
	free(fa3);
	fclose(fp1);			
	fclose(fp3);
}

double train_rbm(int index){
	int i,j,p;
	p = index*BATCH_SIZE; 
	double cost ;
	for(i= 0 ; i<BATCH_SIZE ; i++,p++){
		for(j=0 ; j< NUMVIS ; j++){
			Input[i][j] = Train_Pixel[p][j];			
		}
	}
	if(index==0){
		cost = rbm(false);
	}else{
		cost = rbm(false);
	}
	return cost;	
}

void test_vhv(){
	int i, j ,n_steps; 	
	for(n_steps=0 ;n_steps<1000 ;n_steps++){
		for(i=0 ; i<BATCH_SIZE ; i++){
			for(j=0 ; j<NUMVIS ;j++){			
				//PCD 的起始为persistent	，后面的调用	
				if(n_steps ==0){
					v0_sample[i][j] = vis_persistent[i][j];
				}else{
					v0_sample[i][j] = v1_sample[i][j];
				}						
				sample_h_given_v();
			}
			for(j=0 ; j<NUMHID ;j++){		
				h0_sample[i][j] = h1_sample[i][j];
				sample_v_given_h();
			}
		}
	}
	//更新persistent
	for(i=0 ; i<BATCH_SIZE ; i++){
		for(j=0 ; j<NUMVIS ; j++){
			vis_persistent[i][j] = v1_sample[i][j];
		}
	}	
}

void main(){
	
	
	load_data();
	initialize_W();
	puts("finished");
	
	int i,test_idx,j;
	int training_epochs = 15;	
	int n_chains = 20;
	int n_samples = 10;
	int epoch, batch_index , n_train_batches ;
	double mean_cost,cost;
	clock_t start_time, end_time, plotting_start, plotting_stop;
	double plotting_time = 0;
	double pretrainint_time;
	
	FILE *fp;
	
	start_time = clock();
	n_train_batches = TRAIN_LENGTH/BATCH_SIZE;
	/*////////////////
	////training rbm//	
	*////////////////
	for(epoch=0 ; epoch<training_epochs ; epoch++){
		mean_cost = 0 ;
		for(batch_index=0 ; batch_index<n_train_batches ; batch_index++){
			mean_cost += train_rbm(batch_index);
			//输出权重值
			
			printf("cost is %lf , epoch is %d , batch_index is %d\n", mean_cost/(batch_index+1), epoch, batch_index);
		}
		fp = fopen("weight_epoch.txt","wr");
				for(i=0 ;i<NUMVIS  ;i++){
					for(j=0 ;j<NUMHID ;j++){
						fprintf(fp ,"%lf\n" ,Weight[i][j] );
					}
		}
		printf("Training epoch %d, mean_cost is %lf\n", epoch,mean_cost/n_train_batches);
		//在每次训练之后画图
		plotting_start = clock();
		
		plotting_stop = clock();
		plotting_time += (double)(plotting_stop - plotting_start)/CLOCKS_PER_SEC;
	
	}	
	end_time = clock();
	pretrainint_time = (double)(end_time - start_time)/CLOCKS_PER_SEC - plotting_time;
	printf("Training took %f minutes\n", pretrainint_time/60.0);
	
	/*/////////////////////////
	////sampling from the rbm//	
	*//////////////////////////
	test_idx = rand()/(RAND_MAX/(TEST_LENGTH-n_chains)+1);
	for(i=0 ;i<BATCH_SIZE ;i++,test_idx++){
		for(j=0 ;j<NUMVIS ;j++){
			vis_persistent[i][j] = Test_Pixel[test_idx][j];
		}		
	}
	for(i=0 ;i<n_samples ;i++){
		//测试集进行1000次训练马尔可夫链
		test_vhv();		
	}	
}

















