#include <iostream>
#include <fstream>
using namespace std;

int main()
{
	FILE *pf;
	float *data = new float[200];
	pf = fopen("show.bin", "rb");
	fseek (pf , 0, SEEK_SET);
	for(int i = 0; i < 200; i++)
	{   
			float buffer;
			fread(&buffer,1,4,pf);
			data[i] = buffer;
			cout << buffer << endl;
			fseek (pf, 4, SEEK_CUR);
	}   
	fclose(pf);
	return 0;
}


