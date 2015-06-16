/*************************************************************************
	> File Name: init_and_end.cu
	> Author: DB
	> Mail: dongboyaojiayou@163.com
	> Created Time: Thu 23 Apr 2015 05:19:33 PM CST
 ************************************************************************/

__global__ void parameter_set4search(int xlen, int ylen[],const int n1)
{
    const int tid=blockIdx.x* blockDim.x * blockDim.y
                  +threadIdx.y* blockDim.x+threadIdx.x;
    if(tid<n1)
    {


        float dcu0=4.25;
        float d0;
        int Lnorm= xlen<=ylen[tid]?xlen:ylen[tid];
        if(Lnorm<=19)
        {
            d0=0.168;
        }
        else
        {
            d0=(1.24*pow((Lnorm*1.0-15), 1.0/3)-1.8);
        }
        float D0_MIN=d0+0.8;
        d0=D0_MIN;

        float d0_search=d0;
        if(d0_search>8) d0_search=8;
        if(d0_search<4.5) d0_search=4.5;
        float score_d8=1.5*pow(Lnorm*1.0, 0.3)+3.5;

        dLnorm[tid]=Lnorm;
        ddcu0[tid]=dcu0;
        dd0[tid]=d0;
        dD0_MIN[tid]=D0_MIN;
        dd0_search[tid]=d0_search;
        dscore_d8[tid]=score_d8;

        dtmscore[tid]=-1.0;
        dtmscore2[tid]=-1.0;
    }
    //cuPrintf(" %d %d %f %f %f %f %f\n",
    //tid,Lnorm,dcu0/10,d0/10,D0_MIN/10,d0_search/10,score_d8/10);
}


__global__ void copytocpu(float s1[],float s2[],const int n1)
{
    const int tid=blockIdx.x* blockDim.x * blockDim.y
                  +threadIdx.y* blockDim.x+threadIdx.x;
    if(tid<n1)
    {
        s1[tid]=dtmscore[tid];
        s2[tid]=dtmscore2[tid];
        //printf("%f \n",dtmscore[tid]);
    }
}
