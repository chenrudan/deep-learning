#include <iostream>
#include <fstream>
using namespace std;

int main()
{
    FILE *pf;
    pf = fopen("../../crd/cots/binaryfile/input55.bin", "rb");
/*    fseek(pf, 0, SEEK_SET);
    fseek(pf, 0, SEEK_END);
    cout << ftell(pf) << endl;
    */
    for(int pos = 0, k = 0; pos < 100; pos++)
    {   
       // fseek (pf , pos*100*4, SEEK_SET);
        for(int i = 0; i < 100; i++)
        {   
            float buffer;
            fread(&buffer,1,4,pf);
            cout << buffer << endl;
            k++;
            fseek (pf, 4, SEEK_CUR);
        }   
    }  
    fclose(pf);


    return 0;
}



