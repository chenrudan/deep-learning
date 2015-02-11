#include <stdio.h>
#include <time.h> 
#include <math.h>
#include <stdlib.h>
#include<string.h>
#include<strings.h>

#define TRAIN_LENGTH 50000
#define TEST_LENGTH 10000

#define BATCH_SIZE 20

#define bool    _Bool  
#define true    1  
#define false   0

typedef struct rbm_parameter{
    double **W;
    double *vbias;
    double *hbias;
    int n_visible, n_hidden;  //可视节点与隐藏节点   
}rbm_parameter;

typedef struct dataset_blas{
    double *train_pixel;
    double *test_pixel;
}dataset_blas;

void load_data(int, int, dataset_blas, char *, bool);

void main(){
    //先更新参数
    rbm_parameter rm;
    dataset_blas rp;
    
    	
	rm.n_visible = 784;
	rm.n_hidden = 500;	

	char *type_address[] = {"../mlp/train-images.idx3-ubyte","../mlp/t10k-images.idx3-ubyte"};
	rp.train_pixel= (double*)malloc(TRAIN_LENGTH*rm.n_visible * sizeof(double*));
    rp.test_pixel = (double*)malloc(TEST_LENGTH*rm.n_visible * sizeof(double*));
	load_data( TRAIN_LENGTH, rm.n_visible, rp, type_address[0], true);
	load_data( TEST_LENGTH, rm.n_visible, rp, type_address[1], false);
	printf("%lf", rp.train_pixel[0]);
	
	free(rp.train_pixel);
	free(rp.test_pixel);

}

void load_data(int row, int column, dataset_blas rp, char *type_address, bool label  )
{  
	int i, j, startPos;		
	unsigned char *fa;
	FILE *fp;
	int length = row*column; 	
	fa = (char *)malloc(length*sizeof(char));
	
	if (((fp = fopen(type_address, "r")) == NULL)&& fa == NULL)
		printf("cannot open");
	else{
		//按一个字节读取文件
   		fread(fa ,1 ,length ,fp);
		// input image
//		startPos = 16;
		if(label){
		    
		    for(i=0; i<length; i++){
//		        for(j=0; j<column; j++){ 
			        rp.train_pixel[i] = ((int)fa[i+16])/255.0;
//			        startPos++;					
//		        }				
	        }
		}else{
		    
		    for(i=0; i<length; i++){
//			    for(j=0; j<column; j++){ 
				    rp.test_pixel[i] = ((int)fa[i+16])/255.0;
//				    startPos++;					
//			    }				
		    }
		}							
	}	
	free(fa);
	fclose(fp);
}
