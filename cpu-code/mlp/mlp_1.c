#include <stdio.h>
#include <malloc.h>


#define TRAIN_LENGTH 60000
#define VALID_LENGTH 10000 
#define TEST_LENGTH 10000

#define PIXEL_LENGTH 784
#define TRAIN_TEP_LENGTH 500000000
#define VT_TEP_LENGTH 8000000

// the bool varible
#define bool    _Bool  
#define true    1  
#define false   0  

  
double train_pixel[TRAIN_LENGTH][PIXEL_LENGTH];
double valid_pixel[VALID_LENGTH][PIXEL_LENGTH];
double test_pixel[TEST_LENGTH][PIXEL_LENGTH];

double train_label[TRAIN_LENGTH][1];
double valid_label[VALID_LENGTH][1];
double test_label[TEST_LENGTH][1];





//statement of the functions    
void load_data(int, int, double ** , char * ,bool );
void initialize(); 


//处理输入，给出输出，权重值不在此处改变
void layer_in_out(){
	

}


//训练模型，测试模型,bool变量，true需要修改权重，false不用，求差错函数，求似然函数
//void build_modle( struct ,int , bool type){
	

//}




void load_data(int row, int column, double **elemnt, char *type_address, bool type )
{  
	int i, j, startPos;
	double divider;		
	char *fa;
	FILE *fp;
	int length = row*column; 	
	fa = (char *)malloc(length*sizeof(char));
	if (((fp = fopen(type_address, "r")) == NULL)&& fa == NULL)
		printf("cannot open");
	else{
		//按一个字节读取文件
   		fread(fa ,1 ,length ,fp);
		switch(type){ 			
			case true:	 // input image
				startPos = 8;
				divider = 255.0	;
				break;
			case false: //input label
				startPos = 4;
				divider = 1.0;	
				break;								
		}
		for(i=0; i<row; i++){
			for(j=0; j<column; j++){ 
			//bring the input image value into 0~1
				elemnt[i][j] = ((int)fa[startPos])/divider;
			//	printf("%c\n", fa[k]);
				startPos++;					
			}				
		}		
	}	
}
void initialize_dataset(){
        //initialize 一维指针数组用于传递二维数组
	char *type_address[] = {"train-images.idx3-ubyte","train-labels.idx1-ubyte","t10k-images.idx3-ubyte","t10k-labels.idx1-ubyte"}; 
	int i,j,k;	
	double *pa1[TRAIN_LENGTH]; 
	double *pa2[TRAIN_LENGTH];
	double *pa3[TEST_LENGTH];
	double *pa4[TEST_LENGTH];
	double **dpa1;    
	double **dpa2;
	double **dpa3;
	double **dpa4;
	for(i=0 ; i<TRAIN_LENGTH ; i++){
		pa1[i] = train_pixel[i];
		pa2[i] = train_label[i];
	}
	for(i=0 ; i<TEST_LENGTH ; i++){
		pa3[i] = test_pixel[i];
		pa4[i] = test_label[i];
	}
	dpa1 = pa1;
	dpa2 = pa2;
	dpa3 = pa3;
	dpa4 = pa4;
	load_data( TRAIN_LENGTH, PIXEL_LENGTH, dpa1, type_address[0], true);
	load_data( TRAIN_LENGTH, 1, dpa2, type_address[1], false);
	load_data( TEST_LENGTH, PIXEL_LENGTH, dpa3, type_address[2], true);
	load_data( TEST_LENGTH, 1, dpa4, type_address[3], false);
	
	for(i= TRAIN_LENGTH-VALID_LENGTH, k =0; i<TRAIN_LENGTH; i++,k++){
			for(j=0; j<PIXEL_LENGTH; j++){ 
			//bring the input image value into 0~1
				valid_pixel[k][j] = train_pixel[i][j]; 
			//	printf("%c\n", fa[k]);
			}
			valid_label[k][1] = train_label[i][1];				
	}
	
	
}
void main(){
	initialize_dataset();
	
}










