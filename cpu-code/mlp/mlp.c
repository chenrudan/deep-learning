#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#define NUMPAT 10000
#define NUMIN  784
#define NUMHID 500
#define NUMOUT 10
#define BATCH_SIZE 20

#define LENGTH 1
#define SIZE2 50000000
#define SIZE3 60000

#define test_length 10000
#define train_length 50000
#define validation_length 10000

#define bool    _Bool  
#define true    1  
#define false   0  


int train_label[train_length]={0};
int validation_label[validation_length]={0};
int test_label[test_length]={0};


double train_image[train_length][NUMIN];
double validation_image[validation_length][NUMIN];
double test_image[test_length][NUMIN];

unsigned char buf[SIZE2]={0};

double L1_reg = 0.00;
double L2_reg = 0.0001;

double SumH[BATCH_SIZE][NUMHID], WeightIH[NUMIN+1][NUMHID], Hidden[BATCH_SIZE][NUMHID];
double SumO[BATCH_SIZE][NUMOUT], WeightHO[NUMHID+1][NUMOUT],Output[BATCH_SIZE][NUMOUT];
double DeltaO[BATCH_SIZE][NUMOUT], SumDOW[NUMHID], DeltaH[BATCH_SIZE][NUMHID];
double DeltaWeightIH[NUMIN+1][NUMHID], DeltaWeightHO[NUMHID+1][NUMOUT];

bool done_looping = false;
double learning_rate=0.01;

void initalize_w();
double train_model(int minibatch_index);
void update_parameters(int minibatch_index);
double validate_model();
double test_model();
double mlp_model(int i,int index);
double get_L1();
double get_L2_sqr();
double logRegressionLayer(int i,int index);
void hiddenLayer(int i,int index);
void loaddata();
void loaddata_label(char* address,int length);
void loaddata_image(char* address,int length);

main(){
	int patience = 10000;
	int patience_increase =2;
	double improvement_threshold = 0.995;
	
	
	int n_epoches = 1000;
	int batch_size =BATCH_SIZE;
	double best_validation_loss=1.0;
	double this_validation_loss=0.0;
	double test_score =0.0;
	double minibatch_avg_cost =0.0;
	int best_iter = 0;
	int iter =0;
	int epoch = 0;
	int minibatch_index;
	 
	int n_train_batches = train_length/batch_size;
	int validation_frequency = (n_train_batches< patience/2)?n_train_batches:(patience/2);

	loaddata();
	initalize_w();
	
	while(epoch<n_epoches&&!done_looping){
		epoch +=1;
		for(minibatch_index=0;minibatch_index<n_train_batches;minibatch_index++){
			minibatch_avg_cost = train_model(minibatch_index);
			//printf("epoch:%d   minibatch_index:%d  train_cost:%lf \n",epoch-1,minibatch_index,minibatch_avg_cost);	
			iter = (epoch -1)*n_train_batches+minibatch_index;
			if((iter+1)%validation_frequency==0){
				this_validation_loss = validate_model();
				printf("epoch %d ,validation error %lf %% \n",epoch-1,this_validation_loss*100);
				if(this_validation_loss<best_validation_loss){
					if(this_validation_loss<best_validation_loss*improvement_threshold){
						patience = patience> iter*patience_increase?patience:iter*patience_increase;					
					}
					best_validation_loss = this_validation_loss;
					best_iter = iter;
					
					//test it on the test set
					test_score = test_model();
					printf("    epoch %d, test error of best model %lf %% \n",epoch-1,test_score*100);				
				}
							
			}
			if(patience <= iter){
				done_looping = true;
				break;			
			}	
		}
	}
	printf("Optimization complete. Best validation score of %lf %% \n    obtained at interation %d,with the test performance %lf %% \n",best_validation_loss*100,best_iter+1,test_score*100 );
}

void initalize_w(){
	bzero(WeightHO, sizeof(WeightHO));
	int i,j,k;
	// double low = -4*sqrt(6.0/(NUMIN+NUMHID));
	//double high = 4*sqrt(6.0/(NUMIN+NUMHID));
	double low = -sqrt(6.0/(NUMIN+NUMHID));
	double high = sqrt(6.0/(NUMIN+NUMHID));
	for(j=0;j<NUMHID;j++){
		WeightIH[0][j] =0.0;	
	}
	for(i=1;i<NUMIN+1;i++){
		for(j=0;j<NUMHID;j++){
			k=rand()%200;
			if(k<100){
				WeightIH[i] [j] =(k/100.0)*low;		
			}
			else{
				WeightIH[i] [j] =(k-100)/100.0*high;			
			}
		}
	}
	
	
}

double train_model(int minibatch_index){
	double train_loss = 0.0;
	train_loss = mlp_model(minibatch_index,0);
	update_parameters(minibatch_index);
	return train_loss;
}

void update_parameters(int minibatch_index){
	int p,i,j,k;
	int index=minibatch_index*BATCH_SIZE;
	
	//get delta
	for(p=0;p<BATCH_SIZE;p++){
		for(k=0;k<NUMOUT;k++){
        		if(k==train_label[index+p]){
           			DeltaO[p] [k] = Output[p] [k] -1;
        		}
        		else{
            			DeltaO[p] [k] = Output[p] [k] ;
        		}
    		}
		for( j = 0 ; j < NUMHID ; j++ ) {    
        		SumDOW[j]  = 0.0 ;
        		for( k = 0 ; k < NUMOUT ; k++ ) {
            			SumDOW[j]  += WeightHO[j+1][k]  * DeltaO[p] [k]  ;
        		}
        		//DeltaH[p] [j]  = SumDOW[j]  * Hidden[p] [j]  * (1.0 - Hidden[p] [j] ) ;
			DeltaH[p] [j] =SumDOW[j] *(1.0-Hidden[p] [j] *Hidden[p] [j] );
    		}
		
	}
	
	for(k=0;k<NUMOUT;k++){
		double sum=0.0;
		double sum_Delta_hidden ;
		
		for(p=0;p<BATCH_SIZE;p++){
			sum+=DeltaO[p] [k] ;			
		}
		DeltaWeightHO[0][k] =sum/BATCH_SIZE;
		WeightHO[0][k] =WeightHO[0][k] -learning_rate*DeltaWeightHO[0][k] ;
		for(j=0;j<NUMHID;j++){
			sum_Delta_hidden=0.0;
			for(p=0;p<BATCH_SIZE;p++){
				sum_Delta_hidden+=DeltaO[p] [k] *Hidden[p] [j] ;			
			}
			DeltaWeightHO[j+1][k] =sum_Delta_hidden/BATCH_SIZE;
			WeightHO[j+1][k] =WeightHO[j+1][k] -learning_rate*DeltaWeightHO[j+1][k] ;		
		}	
	}
	
	for(j=0;j<NUMHID;j++){
		double sum=0.0;
		double sum_Delta_input;
		for(p=0;p<BATCH_SIZE;p++){
			sum+=DeltaH[p] [j] ;			
		}
		DeltaWeightIH[0][j] =sum/BATCH_SIZE;
		WeightIH[0][j] =WeightIH[0][j] -learning_rate*DeltaWeightIH[0][j] ;
		for(i=0;i<NUMIN;i++){
			sum_Delta_input =0.0;
			for(p=0;p<BATCH_SIZE;p++){
				sum_Delta_input+=DeltaH[p] [j] *train_image[index+p][i] ;			
			}
			DeltaWeightIH[i+1][j] =sum_Delta_input/BATCH_SIZE;
			WeightIH[i+1][j] =WeightIH[i+1][j] -learning_rate*DeltaWeightIH[i+1][j] ;		
		}	
	}
	
}

double validate_model(){
	int i;
	int n_validate_batches = validation_length/BATCH_SIZE;
	double validate_loss =0.0;
	for(i=0;i<n_validate_batches;i++){
		validate_loss+=mlp_model(i,1);	
	}
	validate_loss = validate_loss/n_validate_batches;
	return validate_loss;
}

double test_model(){
	int i;
	int n_test_batches = test_length/BATCH_SIZE;
	double test_loss =0.0;
	for(i=0;i<n_test_batches;i++){
		test_loss+=mlp_model(i,2);	
	}
	test_loss = test_loss/n_test_batches;
	return test_loss;
}

double mlp_model(int i,int index){
	double cost;
	double L1,L2_sqr;
	double negative_log_likelihood;
	double errors;
	L1 = get_L1();
	L2_sqr = get_L2_sqr();
	if(index==0){
		negative_log_likelihood = logRegressionLayer(i,index);
		cost = negative_log_likelihood+L1_reg*L1+L2_reg*L2_sqr;
		return cost;
	}
	else{
		errors = logRegressionLayer(i,index);
		return errors;
	}
}

double get_L1(){
	int i,j;
	double L1=0.0;
	for(i=0;i<NUMIN+1;i++){
		for(j=0;j<NUMHID;j++){
			L1 +=fabs(WeightIH[i] [j] );
		}
	}
	for(i=0;i<NUMHID+1;i++){
		for(j=0;j<NUMOUT;j++){
			L1 +=fabs(WeightHO[i] [j] );
		}
	}
	return L1;
}

double get_L2_sqr(){
	int i,j;
	double L2 =0.0;
	for(i=0;i<NUMIN+1;i++){
		for(j=0;j<NUMHID;j++){
			L2 +=pow(WeightIH[i] [j] ,2);
		}
	}
	for(i=0;i<NUMHID+1;i++){
		for(j=0;j<NUMOUT;j++){
			L2 +=pow(WeightHO[i] [j] ,2);
		}
	}
	return L2;
}

double logRegressionLayer(int i,int index){
	hiddenLayer(i,index);
	int p,index_Input,k,j;
	index_Input=i*BATCH_SIZE;
	
	double negative_log = 0.0;
	double errors = 0.0;
	int index_max;
	int label =0;
	for(p=0;p<BATCH_SIZE;p++){
		double softmax_add =0.0;
		for( k = 0 ; k < NUMOUT ; k++ ) {    
        		SumO[p] [k]  = WeightHO[0][k]  ;
        		for( j = 0 ; j < NUMHID ; j++ ) {
            			SumO[p] [k]  += Hidden[p] [j]  * WeightHO[j+1][k]  ;
        		}	
 			softmax_add+=exp(SumO[p] [k] );
   	 	}

		for(k=0;k<NUMOUT;k++){
        		Output[p] [k] =exp(SumO[p] [k] )/softmax_add;
               
    		}
		index_max=get_index_max(Output[p] );
		
		switch(index){
			case 0:
				label = train_label[index_Input+p];
				negative_log -=log(Output[p] [label]);
				//if(index_max!=label){
                		//	errors+=1.0;
        			//}
				break;
			case 1:
				label = validation_label[index_Input+p];
				//negative_log -=log(Output[p] [label]);
				if(index_max!=label){
                			errors+=1.0;
        			}
				break;
			case 2:
				label = test_label[index_Input+p];
				//negative_log -=log(Output[p] [label]);
				if(index_max!=label){
                			errors+=1.0;
        			}	
				break;
			default:
				printf("something wrong with index\n");
				done_looping = true;	
	
		}
	}
	errors = errors/BATCH_SIZE;
	negative_log = negative_log/BATCH_SIZE;
	if(index==0){
		return negative_log;
	}
	else{
		return errors;
	}
}

void hiddenLayer(int i,int index){
	int p,index_Input,j,i1;
	index_Input=i*BATCH_SIZE;
	for(p=0;p<BATCH_SIZE;p++,index_Input++){
		for( j = 0 ; j < NUMHID ; j++ ) {    
        	SumH[p] [j]  = WeightIH[0][j]  ;
        	for( i1 = 0 ; i1 < NUMIN ; i1++ ) {
				switch(index){
					case 0:
						SumH[p] [j]  += train_image[index_Input][i1] * WeightIH[i1+1][j]  ;
						break;
					case 1:
						SumH[p] [j]  += validation_image[index_Input][i1] * WeightIH[i1+1][j]  ;
                				break;
					case 2:
						SumH[p] [j]  += test_image[index_Input][i1] * WeightIH[i1+1][j]  ;
						break;
					default:
						printf("something wrong with index\n");
						done_looping = true;	
				}
            	
        	}
        	//Hidden[p] [j]  = 1.0/(1.0 + exp(-SumH[p] [j] )) ;
   	 	Hidden[p] [j]  = tanh(SumH[p] [j] );
		}
		
	}
}

int get_index_max(double *a){
    double b=a[0];
    int c=0;
    int i;
    for(i=1;i<NUMOUT;i++){
        if(b<a[i] ){
            b=a[i] ;
            c=i;
        }
    }
    return c;
}

void loaddata(){

    
    bzero(test_image, sizeof(test_image));
    bzero(train_image, sizeof(train_image));
    
    
    
    char *address[]={"t10k-labels.idx1-ubyte","train-labels.idx1-ubyte","t10k-images.idx3-ubyte","train-images.idx3-ubyte"};
    loaddata_label(address[0],0);
    loaddata_label(address[1],1);
    loaddata_image(address[2],0);
    loaddata_image(address[3],1);
    //printf("%lf\n",test_image[0][250]);
}

void loaddata_label(char* address,int length){
    FILE *fp;
    int i;
    fp = fopen(address,"r");
    if(fp!=NULL&&fread(buf,1,SIZE2/LENGTH,fp) >=0){
            puts("success");
            if(length==0){
                for(i=0;i<test_length;i++){
                    test_label[i] =(int)buf[i+8];
                }
            }
            else{
                for(i=0;i<SIZE3;i++){
                    if(i<train_length){
                        train_label[i] =(int)buf[i+8];
                    }
                    else{
                        validation_label[i-train_length]=(int)buf[i+8];
                    }
                }
            }
            
       
    }
    else{
        perror("open the file");
    }
    fclose(fp);
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
                        test_image[i] [j] =(int)buf[index++]/255.0;
                    }
                }
        }
        else{
            for(i=0;i<SIZE3;i++){
                if(i<train_length){
                    for(j=0;j<784;j++){
                        train_image[i] [j] =(int)buf[index++]/255.0;
                    }
                }
                else{
                    for(j=0;j<784;j++){
                        validation_image[i-train_length][j] =(int)buf[index++]/255.0;
                    }
               }
           }
       }
        
    }
    else
        perror("open the file");
    fclose(fp);
}
