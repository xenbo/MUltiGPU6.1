#include <sstream>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <cstring>
#include"tmscore/cuPrintf.cuh"
#include"tmscore/cuPrintf.cu"
#define nmax 360	///每个大段分配额576个
#define ss_j 16
#define ss_b 200

//#define l1 180
//#define l2 280
const int l1=500;
const int l2=280;
using namespace std;
struct pdblist
{
    int flag_len;	//段长 90，100，110，120，130......
    float *xyz;
    int *resno;
    int *len;	//pdb 长度
    int *id;	//pdb id号
    int num;	//pdb 个数
};

const int flag_len[ss_j]= {90,100,110,120,130,140,150,160,170,180,200,220,240,260,280,1200}; //16个大段
struct pdblist *plist;
const int NGPU=2;

__device__ int dLnorm[nmax];
__device__ float ddcu0[nmax];
__device__ float dd0[nmax];
__device__ float dD0_MIN[nmax];
__device__ float dd0_search[nmax];
__device__ float dscore_d8[nmax];

__device__ float xtm1[nmax][l2][3],ytm1[nmax][l2][3];
__device__ float u1[nmax][3][3],t1[nmax][3];
//__device__ float val[nmax][l1+1][l2+1];
//__device__ char path[nmax][l1+1][l2+1];
//__device__ float score[nmax][l1+1][l2+1];
__device__ int secx[nmax][l1];
__device__ int secy[nmax][l2];

__device__ float dtmscore[nmax];
__device__ float dtmscore2[nmax];

__device__ int  invmap[nmax][l2+2];
__device__ int invmapbak[nmax][l2+2];
__device__ int invmap2[nmax][8][l2+2];

#include"tmscore/basic.cu"
#include"tmscore/init_and_end.cu"
#include"tmscore/Kabsch.cu"
#include"tmscore/dnw.cu"
#include"tmscore/TMscore8_search2.cu"
#include"tmscore/TMscore8_search3.cu"
#include"tmscore/get_init.cu"
#include"tmscore/get_initial_ss.cu"
#include"tmscore/get_initial_local.cu"
#include"tmscore/get_initial_ssplus.cu"
#include"tmscore/get_initial_fgt.cu"
#include"tmscore/final.cu"
#include"tmscore/DP_iter.cu"


int switch_l(int l)
{
    if (l<=90) {
        return  0;
    }
    if (l<=100) {
        return  1;
    }
    if (l<=110) {
        return  2;
    }
    if (l<=120) {
        return  3;
    }
    if (l<=130) {
        return  4;
    }
    if (l<=140) {
        return  5;
    }
    if (l<=150) {
        return  6;
    }
    if (l<=160) {
        return  7;
    }
    if (l<=170) {
        return  8;
    }
    if (l<=180) {
        return  9;
    }
    if (l<=200) {
        return  10;
    }
    if (l<=220) {
        return  11;
    }
    if (l<=240) {
        return  12;
    }
    if (l<=260) {
        return  13;
    }
    if (l<=280) {
        return  14;
    }
    return 15;
}

void get_xyz(string line, float *x, float *y, float *z, int *no)
{
    char cstr[50];

    strcpy(cstr, (line.substr(30, 8)).c_str());
    sscanf(cstr, "%f", x);

    strcpy(cstr, (line.substr(38, 8)).c_str());
    sscanf(cstr, "%f", y);

    strcpy(cstr, (line.substr(46, 8)).c_str());
    sscanf(cstr, "%f", z);

    strcpy(cstr, (line.substr(22, 4)).c_str());
    sscanf(cstr, "%d", no);
}

int read_PDB(const char *filename, float a[][3], int *resno)
{
    int i=0;
    string line, str;
    string atom ("ATOM ");

    ifstream fin (filename);
    if (fin.is_open())
    {
        while ( fin.good() )
        {
            getline(fin, line);
            if(line.compare(0, atom.length(), atom)==0)
            {
                if( line.compare(12, 4, "CA  ")==0 ||\
                        line.compare(12, 4, " CA ")==0 ||\
                        line.compare(12, 4, "  CA")==0 )
                {
                    if( line.compare(16, 1, " ")==0 ||\
                            line.compare(16, 1, "A")==0 )
                    {
                        get_xyz(line, &a[i][0], &a[i][1], &a[i][2],&resno[i]);
                        i++;
                    }
                }
            }
        }
        fin.close();
    }
    return i;
}


int read_list(char *filelist)
{
    float a[2000][3];
    int b[2000];
    int id=0;
    ifstream fin(filelist);
    if (fin.is_open())
    {
        while (fin.good())
        {   memset(&a[0][0],0,2000*3*sizeof(float));

            string line, str;
            getline(fin, line);
            if(line.length()>0)
            {
                string s=line.substr(0,line.length());
                int l=read_PDB(s.c_str(), a, b); //装进a[][]
                int n=switch_l(l);
                while(plist[n].num>=nmax)	n=n+ss_j;
                memcpy(plist[n].xyz+(plist[n].num*plist[n].flag_len*3),&a[0][0],l*3*sizeof(float));
		memcpy(plist[n].resno+(plist[n].num*plist[n].flag_len),b,l*sizeof(int));
                plist[n].len[plist[n].num]=l;
                plist[n].id[plist[n].num]=id;
                plist[n].num++;
                id++;
            }
        }
        fin.close();
    }

    return id;
}


int main(int argv,char *argc[])
{
    //******************************************************************************************
    //******************************加载pdb 数据*************************************************
    //******************************************************************************************
    float *hscore1[ss_b];
    float *hscore2[ss_b];
    plist=(struct pdblist *)malloc(sizeof(struct pdblist)*ss_b);
    for(int i=0; i<ss_b; i++)
    {
        plist[i].xyz=(float*)malloc(sizeof(float)*nmax*flag_len[i%ss_j]*3);
	plist[i].resno=(int *)malloc(sizeof(int)*nmax*flag_len[i%ss_j]);
        plist[i].id=(int *)malloc(sizeof(int)*nmax);
        plist[i].len=(int *)malloc(sizeof(int)*nmax);
        memset(plist[i].xyz,0,sizeof(float)*nmax*flag_len[i%ss_j]*3);
	memset(plist[i].resno,0,sizeof(int)*nmax*flag_len[i%ss_j]);
	memset(plist[i].len,0,sizeof(int)*nmax);
        memset(plist[i].id,0,sizeof(int)*nmax);
        plist[i].flag_len=flag_len[i%ss_j];
        plist[i].num=0;
        hscore1[i]=(float*)malloc(sizeof(float)*nmax);
        hscore2[i]=(float*)malloc(sizeof(float)*nmax);
    }

    char *p="name.txt";
    const int n2=read_list(p);
    float p1[2000][3];
    int b1[2000];
    const int len1=read_PDB("1.pdb",p1,b1);

    printf("******************pdb1 %d *********************\n",len1);
    printf("******************pdb0 %d *********************\n",n2);

    //********************************************************************************************
    //******************************gpu处理 数据***************************************************
    //********************************************************************************************
    dim3 b_(32,1);
    dim3 b_2(4,8);

    cudaStream_t *streams=(cudaStream_t*)malloc(sizeof(cudaStream_t)*NGPU);
    double t1 = omp_get_wtime();
    #pragma omp parallel num_threads(NGPU)
    {
        int ngid = omp_get_thread_num();
        cudaSetDevice(ngid);
        {
            cudaStreamCreate(&(streams[ngid]));
            float *dp1,(*dp21)[3];
            float *dp0,(*dp20)[3];

            float *score1,*score2;
            int *dln,*dresno0,*dresno1;
            float *valline1;
            char  *pathline1;
            float  *dscore1;
            cudaMalloc((void**)&(dp1),sizeof(float)*(len1+3)*3);
            cudaMalloc((void**)&(dp0),sizeof(float)*l2*3*nmax);
	    dp21=(float (*)[3])&(dp1[0]);
            dp20=(float (*)[3])&(dp0[0]);


            cudaMalloc((void**)&dln,sizeof(int)*nmax);
	    cudaMalloc((void**)&dresno1,sizeof(int)*l1);
	    cudaMalloc((void**)&dresno0,sizeof(int)*nmax*l2);
            cudaMalloc((void**)&(score1),sizeof(float)*nmax);
            cudaMalloc((void**)&(score2),sizeof(float)*nmax);
            
            cudaMalloc((void**)&(valline1),sizeof(float)*nmax*(len1+1)*(l2+1));
            cudaMalloc((void**)&(pathline1),sizeof(char)*nmax*(len1+1)*(l2+1));
            cudaMalloc((void**)&(dscore1),sizeof(float)*nmax*(len1+1)*(l2+1));

            cudaMemcpyAsync(dp1,&p1[0][0],sizeof(float)*(len1*3),cudaMemcpyHostToDevice,streams[ngid]);//加载目标pdb1数据
	    cudaMemcpyAsync(dresno1,b1,sizeof(int)*(len1),cudaMemcpyHostToDevice,streams[ngid]);//加载目标pdb1数据
            //cudaPrintfInit();
            for(int c1=ngid; c1<ss_b; c1=c1+NGPU)//openmp 可以NGPU个批次计算
            {
                if(plist[c1].num>0)
                {
                    const int n1=plist[c1].num;
                    const int ss=plist[c1].flag_len;

                    dim3 g(n1,1);
                    //******************************数据加载到 gpu***************************
                    cudaMemcpyAsync(dp0,plist[c1].xyz,sizeof(float)*(nmax*ss*3),cudaMemcpyHostToDevice,streams[ngid]);
                    cudaMemcpyAsync(dln,plist[c1].len,sizeof(int)*(nmax),cudaMemcpyHostToDevice,streams[ngid]);
		    cudaMemcpyAsync(dresno0,plist[c1].resno,sizeof(int)*nmax*ss,cudaMemcpyHostToDevice,streams[ngid]);

                    //******************************gpu处理 设置参数************************
                    parameter_set4search<<<g,b_,0,streams[ngid]>>>(len1,dln,n1);
                    //******************************gpu处理 数据1***************************
                    get_initial2<<<g,b_,0,streams[ngid]>>>(dp21,dp20,len1,dln,ss,score1);
                    detailed_search1<<<g,b_2,0,streams[ngid]>>>(dp21,dp20,len1,dln,40,8,ss,score1);
                    GP_iter1<<<g,b_2,0,streams[ngid]>>>(dp21,dp20,len1,dln,0,2,30,ss,valline1,pathline1,score1);

                    //******************************gpu处理 数据2***************************
                    get_initial_ss2<<<g,b_,0,streams[ngid]>>>(dp21,dp20,len1,dln,ss,valline1,pathline1,score1);
                    detailed_search2<<<g,b_2,0,streams[ngid]>>>(dp21,dp20,len1,dln,40,8,ss,score1);
                    GP_iter2<<<g,b_2,0,streams[ngid]>>>(dp21,dp20,len1,dln,0,2,30,ss,valline1,pathline1,score1);

                    //******************************gpu处理 数据3***************************
                    get_initial_local2<<<g,b_,0,streams[ngid]>>>(dp21,dp20,len1,dln,ss,valline1,pathline1,score1);
                    detailed_search3<<<g,b_2,0,streams[ngid]>>>(dp21,dp20,len1,dln,40,8,ss,score1);
                    GP_iter3<<<g,b_2,0,streams[ngid]>>>(dp21,dp20,len1,dln,0,2,2,ss,valline1,pathline1,score1);

                    //******************************gpu处理 数据4***************************
                    get_initial_ssplus2<<<g,b_,0,streams[ngid]>>>(dp21,dp20,len1,dln,ss,dscore1,valline1,pathline1,score1);
                    detailed_search4<<<g,b_2,0,streams[ngid]>>>(dp21,dp20,len1,dln,40,8,ss,score1);
                    GP_iter4<<<g,b_2,0,streams[ngid]>>>(dp21,dp20,len1,dln,0,2,30,ss,valline1,pathline1,score1);

                    //******************************gpu处理 数据5***************************
                    get_initial_fgt2<<<g,b_,0,streams[ngid]>>>(dp21,dp20,dresno1,dresno0,len1,dln,ss,score1);
                    detailed_search5<<<g,b_2,0,streams[ngid]>>>(dp21,dp20,len1,dln,40,8,ss,score1);
                    GP_iter5<<<g,b_2,0,streams[ngid]>>>(dp21,dp20,len1,dln,1,2,2,ss,valline1,pathline1,score1);

                    //******************************gpu处理 数据6***************************
	            detailed_search6<<<g,b_2,0,streams[ngid]>>>(dp21,dp20,len1,dln,1,8,ss,score1);
	            Gfinal_TMscore8_search<<<g,b_2,0,streams[ngid]>>>(dp21,dp20,len1,dln,ss,score1);
                    
		    //******************************gpu 数据拷回***************************
                    copytocpu<<<g,b_2,0,streams[ngid]>>>(score1,score2,n1);
                    cudaMemcpyAsync(hscore1[c1],score1,sizeof(float)*(n1),cudaMemcpyDeviceToHost,streams[ngid]);
                    cudaMemcpyAsync(hscore2[c1],score2,sizeof(float)*(n1),cudaMemcpyDeviceToHost,streams[ngid]);


                }
            }
            //cudaPrintfDisplay(stdout,true);
            //cudaPrintfEnd();
            //cudaDeviceSynchronize();
            //****************************** 数据 释放**********************************
            cudaFree(dp0);
            cudaFree(dp1);
            cudaFree(score1);
            cudaStreamDestroy(streams[ngid]);
        }

    }



    double t2 = omp_get_wtime();
    /*
    	for(int i=1; i<=(n1/32*32)*l2; i++)
        {
            printf("%.0f ",hscore1[i-1]);
    		if(i%l2==0)printf("\n\n");
        }
     */
    //********************************************************************************************
    //******************************数据  输出***************************************************
    //********************************************************************************************
    char c0[6];
    for(int i=0; i<ss_b; i++)
    {
        if(plist[i].num>0)
        {
            for(int j=0; j<plist[i].num; j++)
            {
                int id=plist[i].id[j];
                printf("%s%d  %.6f  %.6f\n",get50(id,c0),id,hscore1[i][j],hscore2[i][j]);
            }
        }
        free(hscore1[i]);
        free(hscore2[i]);
    }

    printf("time: %f ms\n",t2-t1);
    cudaDeviceReset();
}
