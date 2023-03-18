#include<bits/stdc++.h>
#include <unistd.h>
using namespace std;

bool is_finish(){
//	freopen("is_finish.txt","r",stdin);
	FILE * fp = fopen( "is_finish.txt" , "r" );
	rewind(fp);
	int x;
	fscanf(fp,"%d",&x);
	fclose(fp);
	return (x==0);
}
double _lr[5] = {0.01,0.005,0.001,0.0005,0.0001};
double _dropout[3] = {0.1,0.3,0.5};
double _weight_decay[4] = {0,0.0003,0.0002,0.0001};
int use[100]={};

int main(){
	
	for(int i=0;i<5;i++)
	for(int j=0;j<3;j++)
	for(int k=0;k<4;k++){
		double lr = _lr[i];
		double dropout = _dropout[j];
		double weight_decay = _weight_decay[k];
		
//		int tmp = i*100+j*10+k;
//		bool f=true;
//		for(int t=0;t<9;t++)
//			if(use[t]==tmp){
//				f=false;
//				break;
//			}
//		if(f==false) continue;
		
		FILE * fp = fopen( "parameter.txt" , "w" );
		rewind(fp);
		fprintf(fp,"%lf %lf %lf",lr,dropout,weight_decay);
		fclose(fp);
		system("C:/Users/maobochao/AppData/Local/Programs/Python/Python38/python.exe c:/Users/maobochao/Desktop/EOJÍÆ¼öÏµÍ³/Program/run.py");
//		sleep(10);
		while(is_finish()) sleep(5),cerr<<"runing!!!!!!!!!!!!!!!!!!!!!!"<<endl;
	}
	
	
	return 0;
} 
