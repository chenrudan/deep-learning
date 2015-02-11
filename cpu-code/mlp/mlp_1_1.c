#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>


#define TRAIN_LENGTH 50000
#define VALID_LENGTH 10000 
#define TEST_LENGTH 10000

//输入节点有500个，隐藏层节点有500个，输出节点有10个
#define NUMIN 784
#define NUMHID 500
#define NUMOUT 10

//epochs是全部训练集循环次数， 一次性批处理个数为20 ，
#define N_Epochs 1000
#define BATCH_SIZE 20

// the bool varible
#define bool    _Bool  
#define true    1  
#define false   0  
	
double Train_Pixel[TRAIN_LENGTH][NUMIN];
double Valid_Pixel[VALID_LENGTH][NUMIN];
double Test_Pixel[TEST_LENGTH][NUMIN];

double Train_Label[TRAIN_LENGTH];
double Valid_Label[VALID_LENGTH];
double Test_Label[TEST_LENGTH];

	

double Log_Likelihood;

int BatchIndex;
double learning_rate=0.01;

//hidden表示隐藏层的输出,激励函数的输出，sum表示隐藏层每个节点的输入，作为激励函数的输入
double SumH[BATCH_SIZE][NUMHID], WeightIH[NUMIN+1][NUMHID], Hidden[BATCH_SIZE][NUMHID];
double SumO[BATCH_SIZE][NUMOUT], WeightHO[NUMHID+1][NUMOUT], Output[BATCH_SIZE][NUMOUT];
double DeltaO[NUMOUT], SumDOW[NUMHID], DeltaH[NUMHID];
double DeltaWeightIH[NUMIN+1][NUMHID], DeltaWeightHO[NUMHID+1][NUMOUT];
double DeltaH_array[BATCH_SIZE][NUMHID], DeltaO_array[BATCH_SIZE][NUMOUT];

double Input[BATCH_SIZE][NUMIN];
double Target[BATCH_SIZE];



//statement of the functions    
void load_data();
double layer_in_out(int , bool );
int choose_max(double *);
void initialize_weight();
double train_model(int );
double valid_model();
double test_model();
double L1();
double L2();


//处理输入，给出输出, 求差错函数，求似然函数,假如输入为训练集，那么修改权重
double layer_in_out(int BatchIndex , bool Input_Type){
	int i,j,k, max_index, np , p,index;
	//index = BatchIndex*BATCH_SIZE;
	double Error = 0;
	double Log_Output = 0.0; 
	
	

	for(p=0; p<BATCH_SIZE ; p++){
		//weightIH是749*500
		double softmax_add = 0;
		int label;
		
		for(j = 0 ; j < NUMHID ;j++){
			SumH[p][j] = WeightIH[0][j] ;
			for(i=0 ; i< NUMIN ; i++){
				SumH[p][j] += Input[p][i] * WeightIH[i+1][j] ;
			}
			//Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j])) ;
			Hidden[p][j]  = tanh(SumH[p][j]);
		}
		for( k = 0 ; k < NUMOUT ; k++ ) {    /* compute output unit activations */
			SumO[p][k] = WeightHO[0][k] ;
			for( j = 0 ; j < NUMHID ; j++ ) {
				SumO[p][k] += Hidden[p][j] * WeightHO[j+1][k] ;
			}
			//Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k])) ; 
			softmax_add+=exp(SumO[p][k]);    
       		}                
    		for(k=0;k<NUMOUT;k++){
			Output[p][k] =exp(SumO[p][k] )/softmax_add;
		       
		}
		label = Target[p];
		Log_Output-=log(Output[p][label]);
		
		//计算delta  k,通过10个输出与该标签进行对比,假如相同，那么target就应该等于1，其他的为0，[0,0,0,1,0,0,0,0,0,0]这种
	    	for(k=0;k<NUMOUT;k++){
			if(k==Target[p]){
   				DeltaO_array[p][k] = Output[p][k] -1;
			}
			else{
				DeltaO_array[p][k] = Output[p][k] ;
			}
		}    
				
		max_index = choose_max(Output[p]);
		//Log_Output += -log(Output[p][max_index]) ;
		if(max_index != Target[p]){
			Error += 1 ;	
		}
								

			//输入为训练集
		if(Input_Type == true){		
			for( j = 0 ; j < NUMHID ; j++ ) {    /* 'back-propagate' errors to hidden layer */
			
				SumDOW[j] = 0.0 ;
				for( k = 0 ; k < NUMOUT ; k++ ) {
		    			SumDOW[j] += WeightHO[j+1][k] * DeltaO_array[p][k] ;
				}
		       	DeltaH_array[p][j] = SumDOW[j] * (1.0 - Hidden[p][j] * Hidden[p][j]) ;
	    	}		    	
		}
	
	}
	
	Log_Output=Log_Output/BATCH_SIZE;
//	printf("BatchIndex:%d        Log_Output:%lf\n",BatchIndex,Log_Output);	
	
	if(Input_Type == true){
	
		double sum ;
		for( j = 0 ; j < NUMHID ; j++ ) {     /* update weights WeightIH */
	   		//修改b的值
			DeltaH[j]=0;
			
	   		for(p=0 ; p<BATCH_SIZE ; p++){
				DeltaH[j] += DeltaH_array[p][j];
			}
			DeltaWeightIH[0][j] = DeltaH[j]/BATCH_SIZE ;
			WeightIH[0][j] = WeightIH[0][j] -learning_rate*DeltaWeightIH[0][j] ;
			for( i = 0 ; i < NUMIN ; i++ ) { 
				sum =0;
				for(p =0; p<BATCH_SIZE;p++){
					sum += Input[p][i] * DeltaH_array[p][j];
				}
		    		DeltaWeightIH[i+1][j] =  sum/BATCH_SIZE ;
		    		WeightIH[i+1][j] = WeightIH[i+1][j] -learning_rate*DeltaWeightIH[i+1][j] ;
			}
			
    		}
	    	for( k = 0 ; k < NUMOUT ; k ++ ) {    /* update weights WeightHO */
			DeltaO[k]=0;
			
	    		for(p= 0 ; p<BATCH_SIZE ; p++){
					DeltaO[k] += DeltaO_array[p][k];				
			}
			DeltaWeightHO[0][k] = DeltaO[k]/BATCH_SIZE  ;
			WeightHO[0][k] = WeightHO[0][k] -learning_rate*DeltaWeightHO[0][k] ;
			for( j = 0 ; j < NUMHID ; j++ ) {
				sum =0;
				for( p=0 ;p<BATCH_SIZE ; p++){
					sum +=  Hidden[p][j] * DeltaO_array[p][k];  
				}
		    		DeltaWeightHO[j+1][k] =  sum/BATCH_SIZE  ;
		    		WeightHO[j+1][k] = WeightHO[j+1][k] -learning_rate*DeltaWeightHO[j+1][k] ;
			}	
    		}
    	//return Error;
	    	return Log_Output;
	}
	else{	
		return Error;
	}
}



//返回一组数据中最大值的脚标
int choose_max(double num_array[NUMOUT]){
	int max_index , i;
	max_index = 0;
	double max_value = num_array[0];	
	for(i=1 ; i<NUMOUT; i++){
		if(max_value < num_array[i]){
			max_index = i;
			max_value = num_array[i];
		}	
	}
	return max_index;
}



//训练模型， 修改权重
double train_model(int BatchIndex){
	int i,j ;
	double train_logistic_log = 0.0 ;
	double cost =0.0;
	int index = BatchIndex*BATCH_SIZE;
	for(i= 0 ; i<BATCH_SIZE ; i++,index++){
		for(j=0 ; j< NUMIN ; j++){
			Input[i][j] = Train_Pixel[index][j];
			
		}
		Target[i] = Train_Label[index];
	}
	//训练模型返回值是似然函数对数
	train_logistic_log = layer_in_out(BatchIndex, true);
	//printf("%lf \n", train_logistic_log);
	//cost = train_logistic_log + L1() + L2();
	//return cost;
	return train_logistic_log;
}

double L1(){
	int i,j,k ;
	double L1 =0.0;
	for(i= 0; i<NUMIN; i++){
		for(j = 0; j<NUMHID; j++){
			L1 += fabs(WeightIH[i][j]) ;
		}
	}
	for(j= 0; j<NUMHID; j++){
		for(k = 0; k<NUMOUT; k++){
			L1 += fabs(WeightHO[j][k]) ;
		}
	}
	return L1;	
}

double L2(){
	int i,j,k ;
	double L2 =0.0;
	for(i= 0; i<NUMIN; i++){
		for(j = 0; j<NUMHID; j++){
			L2 += pow(WeightIH[i][j], 2) ;
		}
	}
	for(j= 0; j<NUMHID; j++){
		for(k = 0; k<NUMOUT; k++){
			L2 += pow(WeightHO[j][k], 2) ;
		}
	}
	return L2;	
}


double valid_model(){
	int i,j,m ;
	double valid_loss = 0.0;
	int index= 0;
	for(m=0 ;m <VALID_LENGTH/BATCH_SIZE ;m++){
		for(i=0 ; i<BATCH_SIZE; i++,index++){		
			for(j=0 ; j<NUMIN ; j++){		
				Input[i][j] = Valid_Pixel[index][j];
			}
			Target[i] = Valid_Label[index];
		}
		//返回值是error	
		valid_loss += layer_in_out(m, false);
	}
	valid_loss = valid_loss/VALID_LENGTH;
	return valid_loss;
}

double test_model(){
	int i,j,m ;
	double test_loss = 0.0;
	int index = 0;
	for(m=0 ;m <TEST_LENGTH/BATCH_SIZE ;m++){
		for(i=0 ; i<BATCH_SIZE; i++,index++){
			for(j=0 ; j<NUMIN ; j++){			
				Input[i][j] = Test_Pixel[index][j];			
			}
			Target[i] = Test_Label[index];
		}
		test_loss += layer_in_out(m, false);
	}
	test_loss = test_loss/VALID_LENGTH;
	return test_loss;
}



void initialize_weight(){
/*	double eta = 0.5, smallwt = 0.5;
	int i,j,k;
	//产生0～1的随机权重数，权重偏导初始值为0
	srand(1234);
	for( j = 0 ; j < NUMHID ; j++ ) {   
        	for( i = 0 ; i <= NUMIN ; i++ ) { 
       	        	DeltaWeightIH[i][j] = 0.0 ;
            		//WeightIH[i][j] = 2.0 * ( ((double)rand()/(RAND_MAX)) - 0.5 ) * smallwt ;
            		WeightIH[i][j] =  ((double)rand()/(RAND_MAX))  ;
        	}
    	}
    	for( k = 0 ; k < NUMOUT ; k ++ ) {    
        	for( j = 0 ; j <= NUMHID ; j++ ) {
            		DeltaWeightHO[j][k] = 0.0 ;              
            		//WeightHO[j][k] = 2.0 * ( ((double)rand()/(RAND_MAX)) - 0.5 ) * smallwt ;
            		WeightHO[j][k] =  ((double)rand()/(RAND_MAX))  ;
        	}
    	}*/
    	bzero(WeightHO, sizeof(WeightHO));
    	int i,j,k;
	// double low = -4*sqrt(6.0/(NUMIN+NUMHID));
	//double high = 4*sqrt(6.0/(NUMIN+NUMHID));
	double low = -sqrt(6.0/(NUMIN+NUMHID));
	double high = sqrt(6.0/(NUMIN+NUMHID));
	//srand(1234);
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


void load_data(){
	
	int i,j,k, startPos1 = 12, startPos2 =8 ;
	unsigned char *fa1, *fa2, *fa3, *fa4;
	FILE *fp1, *fp2, *fp3, *fp4;	
	fa1 = (char *)malloc(((TRAIN_LENGTH+VALID_LENGTH)*NUMIN)*sizeof(char));
	fa2 = (char *)malloc(((TRAIN_LENGTH+VALID_LENGTH)*NUMIN)*sizeof(char));
	fa3 = (char *)malloc((TEST_LENGTH*NUMIN)*sizeof(char));
	fa4 = (char *)malloc((TEST_LENGTH*NUMIN)*sizeof(char));
	
	fp1 = fopen("train-images.idx3-ubyte", "r");
	fp2 = fopen("train-labels.idx1-ubyte", "r");
       	fp3 = fopen("t10k-images.idx3-ubyte", "r");
	fp4 = fopen("t10k-labels.idx1-ubyte", "r");
	
	fread(fa1 , 1, ((TRAIN_LENGTH+VALID_LENGTH)*NUMIN) , fp1);
	fread(fa2 , 1, TRAIN_LENGTH+VALID_LENGTH , fp2);
	fread(fa3 , 1, (TEST_LENGTH*NUMIN) , fp3);
	fread(fa4 , 1, TEST_LENGTH , fp4);
        //初始化训练集元素
	for(i=0; i<TRAIN_LENGTH ; i++){
		for(j=0; j<NUMIN; j++){ 
		//bring the input image value into 0~1
			Train_Pixel[i][j] = ((int)fa1[startPos1])/255.0;
			startPos1++;					
		}	
		Train_Label[i] = (int)fa2[startPos2];
		startPos2++;			
	}
	//将训练集的后10000元素赋值给有效集
	
	for(  i =0; i<VALID_LENGTH; i++){
		for(j=0; j<NUMIN; j++){ 
		//bring the input image value into 0~1
			Valid_Pixel[i][j] = ((int)fa1[startPos1])/255.0;
		 	startPos1++;
		}
		Valid_Label[i] = (int)fa2[startPos2];
		startPos2++;				
	}
	free(fa1);
	free(fa2);
	//初始化测试集
	startPos1 = 12;
	startPos2 = 8; 
	for(i=0; i<TEST_LENGTH ; i++){
		for(j=0; j<NUMIN; j++){ 
		//bring the input image value into 0~1
			Test_Pixel[i][j] = ((int)fa3[startPos1])/255.0;
			startPos1++;					
		}	
		Test_Label[i] = (int)fa4[startPos2];
		startPos2++;			
	}
	free(fa3);
	free(fa4);
	fclose(fp1);
	fclose(fp2);			
	fclose(fp3);
	fclose(fp4);
}


void main(){
	int p, np , op,  minibatch_index , epoch;
	int iter;
	int best_iter=0;
	int patience = 10000;
	int patience_increase =2;
	double best_validation_loss = 1000000.0;
	bool done_looping = false;
	double minibatch_avg_cost = 0.0;
	double this_validation_loss =0.0;
	double test_loss =0.0;
	double improvement_threshold = 0.995;
	int n_train_batches;
	
	//下载数据以及初始化权重值
	load_data();
	puts("success");
	initialize_weight();
	puts("success");
	
	n_train_batches = TRAIN_LENGTH / BATCH_SIZE;
	int validation_frequency = (n_train_batches< patience/2)?n_train_batches:(patience/2);
	
	
	
	//early stopping
	for(epoch=0 ; (epoch<1000)&&done_looping ==false; epoch++){
		//将输入乱序		
/*		for( p = 0 ; p < TRAIN_LENGTH ; p++ ) {    // 将输入乱序 
            		ranpat[p] = p ;
       	 	}
        	for( p = 0 ; p < TRAIN_LENGTH ; p++) {
      			//      printf("%d\n", rand());
            		np = p + ((double)rand()/(RAND_MAX)) * ( TRAIN_LENGTH + 1 - p ) ;
            		op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;            		
       		}
       		printf("%d", ranpat[6]);
  */     		// 
  		
       		for(minibatch_index=0 ; minibatch_index < n_train_batches ; minibatch_index++  ){
       			minibatch_avg_cost = train_model(minibatch_index);
       		//	printf("   %lf\n", minibatch_avg_cost);
       			//puts("success");
       			iter = (epoch) * n_train_batches + minibatch_index;
      		//	printf("%d \n \n \n", minibatch_index);
       			
       			if((iter + 1)% validation_frequency ==0){
       				this_validation_loss = valid_model() ;   
       				printf("epoch %d ,validation error %lf %%\n",epoch, this_validation_loss*100);
       				
       				if(this_validation_loss <best_validation_loss){
       					if(this_validation_loss <best_validation_loss *improvement_threshold){
       						patience = (patience> iter*patience_increase)?patience:iter*patience_increase;
       					}
       					best_validation_loss = this_validation_loss;
       					best_iter = iter;
       					test_loss = test_model();
       					printf("    epoch %d, test error of best model %lf %% \n",epoch,test_loss*100);
       				}       				
       			}
       			if(patience <= iter){
				done_looping = true;
				break;			
			}
			
       		}
       		//printf("   %lf\n", minibatch_avg_cost);      		
       	}
       	printf("Optimization complete. Best validation score of %lf %% \n    obtained at interation %d,with the test performance %lf %% \n",best_validation_loss*100,best_iter+1,test_loss*100 );	
}










