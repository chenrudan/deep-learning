/// filename: testCin.cpp
//
//
#include <iostream>
#include <vector>
#include <string.h>

using namespace std;

int main(){

	char *in = new char[100];
	char *p = new char[100];
	char mark = ',';
	cin >> in;
	cout << in << endl;
	p = strtok(in, &mark);
	while(p){
		cout << p << endl;
		p = strtok(NULL, &mark);
	}

	return 0;
}
