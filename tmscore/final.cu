
__device__ void parameter_set4final(float len)
{
    float D0_MIN=0.5;
    float d0;
    int Lnorm=len;
    if(Lnorm<=21)
    {
        d0=0.5;
    }
    else
    {
        d0=(1.24*pow((Lnorm*1.0-15), 1.0/3)-1.8);
    }
    if(d0<D0_MIN) d0=D0_MIN;

    float d0_search=d0;
    if(d0_search>8) d0_search=8;
    if(d0_search<4.5) d0_search=4.5;


    dLnorm[blockIdx.x]=Lnorm;
    dd0[blockIdx.x]=d0;
    dD0_MIN[blockIdx.x]=D0_MIN;
    dd0_search[blockIdx.x]=d0_search;
}

__device__ void final_TMscore8_search(
    float x[][3],
    float y[][3],
    int xlen,
    int ylen,
    int map[],
    float *s)
{
    int j=0, k=0;
    float d;
    const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    float  score_d8= dscore_d8[blockIdx.x];
    volatile __shared__ int sk;

    if(tid<3)
    {
        for(j=0; j<ylen; j++)
        {
            int i=map[j];
            if(i>=0&&i<xlen)
            {
                float xt[3];
                transform(t1[blockIdx.x], u1[blockIdx.x], x[i], xt);

                float d0=xt[0]-y[j][0];
                float d1=xt[1]-y[j][1];
                float d2=xt[2]-y[j][2];

                d=d0*d0+d1*d1+d2*d2;

                if(d <= score_d8*score_d8)
                {
                    xtm1[blockIdx.x][k][tid]=x[i][tid];
                    ytm1[blockIdx.x][k][tid]=y[j][tid];
                    k++;
                }

            }
            if(tid==0)
                sk=k;
        }
    }

    if(tid==0)
        parameter_set4final(ylen+0.0);

    float TM1=TMscore8_search3(
                  xtm1[blockIdx.x],
                  ytm1[blockIdx.x],
                  sk,
                  t1[blockIdx.x],
                  u1[blockIdx.x],
                  1,0,0);
    if(tid==0)
        parameter_set4final(xlen+0.0);

    float TM2=TMscore8_search3(
                  xtm1[blockIdx.x],
                  ytm1[blockIdx.x],
                  sk,
                  t1[blockIdx.x],
                  u1[blockIdx.x],
                  1,0,0);
    if(tid==0)
    {

        dtmscore[blockIdx.x]=TM1;
        dtmscore2[blockIdx.x]=TM2;
    }
    __syncthreads();
}



__global__ void Gfinal_TMscore8_search(
    float x[][3],
    float y[][3],
    int xlen,
    int ylen[],
    const int l22,
    float score[])
{
    //if(blockIdx.x==5)
    final_TMscore8_search(
        x,
        &y[blockIdx.x*l22],
        xlen,
        ylen[blockIdx.x],
        invmapbak[blockIdx.x],
        NULL);


    //const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    /*
    if(tid==0&&blockIdx.x==0)
    for(int j=0;j<ylen[blockIdx.x];j++)
    {
    		int i=invmap[blockIdx.x][j];
    		if(i>=0&&i<xlen)
    		{
    			cuPrintf("%d -> %d \n",j,i);
    		}

    }
    */
}

