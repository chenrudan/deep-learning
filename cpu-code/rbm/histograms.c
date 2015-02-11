#include<math.h>

#include<stdio.h>
#include<stdlib.h>
#include<graphics.h>

#define NUMVIS 784
#define LENGTH 49

void main(){
    int *GraphDriver;  //指向显卡类型编号
    int *GraphMode;    //指向显示模式编号
    FILE *fp;
    double data[NUMVIS*LENGTH];
    fp = fopen("weight.txt","r");
    fgets(data,NUMVIS*LENGTH,fp);
    printf("%lf",data[0]);
    fclose(fp);
    
    
    
}
