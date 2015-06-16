__device__ void DP_iter(
    float x[][3],
    float y[][3],
    int x_len,
    int y_len,
    int g1,
    int g2,
    int iteration_max,
    const int l,
    float val[],
    char path[],
    float *s
)
{
    float tmscore=0.0,gap_open[2]= {-0.6, 0};
    int iteration=0, i, j, k;
    float  tmscore_max=0, tmscore_old=0;
    const int tid=threadIdx.y*blockDim.x+threadIdx.x;

    int score_sum_method=8, simplify_step=40;
    volatile __shared__ int sk;
    float d0=dd0[blockIdx.x];
    float d02=d0*d0;

    /*
       if(tid==0)
    	cuPrintf("%f %f %f \n",
    			t1[blockIdx.x][0],
    			t1[blockIdx.x][1],
    			t1[blockIdx.x][2]);

    */

    int g=g1;
    for(; g<g2; g++)
    {
        for(iteration=0; iteration<iteration_max; iteration++)
        {

            DNW(x, y, x_len, y_len,
                t1[blockIdx.x],
                u1[blockIdx.x],
                d02,
                gap_open[g],
                invmap[blockIdx.x],
                l,
                val,
                path);

            k=0;
            if(tid<3)
            {
                for(j=0; j<y_len; j++)
                {
                    i=invmap[blockIdx.x][j];
                    if(i>=0&&i<x_len) //aligned
                    {
                        xtm1[blockIdx.x][k][tid]=x[i][tid];
                        ytm1[blockIdx.x][k][tid]=y[j][tid];
                        k++;
                        //if(tid==0 && iteration==2)
                        //	cuPrintf("%f ->%f\n",y[j][2],x[i][2]);
                    }
                }
                if(tid==0)
                    sk=k;
            }
            //****************************************************
            /*
              if(tid==0)
             {
             	cuPrintf("T: %f  %f  %f  %d  %d  sk:%d tmscore:%f\n",
             			t1[blockIdx.x][0],
             			t1[blockIdx.x][1],
             			t1[blockIdx.x][2],
             			x_len,y_len,sk,tmscore);
             }
            */
            //****************************************************
            //if(tid==0)
            //	printf("  %f  %f  %f  -- TM ylen  %d  %d %f \n",t1[blockIdx.x][0],t1[blockIdx.x][1],t1[blockIdx.x][2],y_len,k,tmscore);

            tmscore= TMscore8_search(
                         xtm1[blockIdx.x],
                         ytm1[blockIdx.x],
                         sk,
                         t1[blockIdx.x],
                         u1[blockIdx.x],
                         simplify_step,
                         score_sum_method,1);


            if(tmscore>tmscore_max)
            {
                tmscore_max=tmscore;

                for(int i=tid; i<y_len; i=i+32)
                {
                    invmap2[blockIdx.x][0][i]=invmap[blockIdx.x][i];
                }
            }
            if(fabs(tmscore_old-tmscore)<0.000001)
            {
                break;
            }
            tmscore_old=tmscore;
        }// for iteration

    }//for gapopen

    if(tid==0)
    {
        dtmscore2[blockIdx.x]=tmscore_max;
        //cuPrintf("========== %d %f\n", sk,tmscore_max);
    }
}



__global__ void GP_iter(
    float x[][3],
    float y[][3],
    int x_len,
    int y_len[],
    int g1,
    int g2,
    int iteration_max,
    const int l22,
    float val[],
    char path[],
    float *s)
{
    DP_iter(x,
            &y[(blockIdx.x)*l22],
            x_len,
            y_len[blockIdx.x],
            g1,g2,iteration_max,l22,
            &val[blockIdx.x*(x_len+1)*(l22+1)],
            &path[blockIdx.x*(x_len+1)*(l22+1)],
            NULL);



    const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    volatile __shared__  int flag;
    if(tid==0)
    {   flag=0;
        if(dtmscore[blockIdx.x]<dtmscore2[blockIdx.x])
            flag=1;
    }

    if(flag)
    {   for(int i=tid; i<l22; i=i+32)
        {
            // invmap[blockIdx.x][i]=invmap2[blockIdx.x][0][i];
            //  s[blockIdx.x*l2+i]=invmap2[blockIdx.x][0][i];
        }
    }
}


__global__ void GP_iter1(
    float x[][3],
    float y[][3],
    int x_len,
    int y_len[],
    int g1,
    int g2,
    int iteration_max,
    const int l22,
    float val[],
    char path[],
    float *s)
{
    DP_iter(x,
            &y[(blockIdx.x)*l22],
            x_len,
            y_len[blockIdx.x],
            g1,g2,iteration_max,l22,
            &val[blockIdx.x*(x_len+1)*(l22+1)],
            &path[blockIdx.x*(x_len+1)*(l22+1)],
            NULL);


    const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    volatile __shared__  int flag;
    if(tid==0)
    {   flag=0;
        if(dtmscore[blockIdx.x]<dtmscore2[blockIdx.x])
        {
            dtmscore[blockIdx.x]=dtmscore2[blockIdx.x];
            flag=1;
        }
    }

    if(flag)
    {   for(int i=tid; i<l22; i=i+32)
        {
            invmapbak[blockIdx.x][i]=invmap2[blockIdx.x][0][i];
            //s[blockIdx.x*l2+i]=invmap2[blockIdx.x][8][i];
        }
    }
}

__global__ void GP_iter2(
    float x[][3],
    float y[][3],
    int x_len,
    int y_len[],
    int g1,
    int g2,
    int iteration_max,
    const int l22,
    float val[],
    char path[],
    float *s)
{


    const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    volatile __shared__  int flag;

    //if(blockIdx.x==0)
    {

        if(tid==0)
            if(dtmscore2[blockIdx.x]>dtmscore[blockIdx.x]*0.2)
                flag=1;

        if(flag)
        {
            DP_iter(x,
                    &y[(blockIdx.x)*l22],
                    x_len,
                    y_len[blockIdx.x],
                    g1,g2,iteration_max,l22,
                    &val[blockIdx.x*(x_len+1)*(l22+1)],
                    &path[blockIdx.x*(x_len+1)*(l22+1)],
                    NULL);
            if(tid==0)
            {
                flag=0;
                if(dtmscore[blockIdx.x]<dtmscore2[blockIdx.x])
                {
                    dtmscore[blockIdx.x]=dtmscore2[blockIdx.x];
                    flag=1;
                }
            }
            if(flag)
            {
                for(int i=tid; i<l22; i=i+32)
                {
                    invmapbak[blockIdx.x][i]=invmap2[blockIdx.x][0][i];
                    //s[blockIdx.x*l2+i]=invmapbak[blockIdx.x][i];
                }
            }
        }
    }
}

__global__ void GP_iter3(
    float x[][3],
    float y[][3],
    int x_len,
    int y_len[],
    int g1,
    int g2,
    int iteration_max,
    const int l22,
    float val[],
    char path[],
    float *s)
{


    const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    volatile __shared__  int flag;

    float  ddcc=0.4;
    if(dLnorm[blockIdx.x] <= 40) ddcc=0.1;
    if(tid==0)
        if(dtmscore2[blockIdx.x]>dtmscore[blockIdx.x]*ddcc)
            flag=1;

    if(flag)
    {
        DP_iter(x,
                &y[(blockIdx.x)*l22],
                x_len,
                y_len[blockIdx.x],
                g1,g2,iteration_max,l22,
                &val[blockIdx.x*(x_len+1)*(l22+1)],
                &path[blockIdx.x*(x_len+1)*(l22+1)],
                NULL);
        if(tid==0)
        {
            flag=0;
            if(dtmscore[blockIdx.x]<dtmscore2[blockIdx.x])
            {
                dtmscore[blockIdx.x]=dtmscore2[blockIdx.x];
                flag=1;
            }
        }
        if(flag)
        {
            for(int i=tid; i<l22; i=i+32)
            {
                invmapbak[blockIdx.x][i]=invmap2[blockIdx.x][0][i];
                // s[blockIdx.x*l2+i]=invmapbak[blockIdx.x][i];
            }
        }
    }
}


__global__ void GP_iter4(
    float x[][3],
    float y[][3],
    int x_len,
    int y_len[],
    int g1,
    int g2,
    int iteration_max,
    const int l22,
    float val[],
    char path[],
    float *s)
{


    const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    volatile __shared__  int flag;

    float  ddcc=0.4;
    if(dLnorm[blockIdx.x] <= 40) ddcc=0.1;
    if(tid==0)
        if(dtmscore2[blockIdx.x]>dtmscore[blockIdx.x]*ddcc)
            flag=1;

    if(flag)
    {
        DP_iter(x,
                &y[(blockIdx.x)*l22],
                x_len,
                y_len[blockIdx.x],
                g1,g2,iteration_max,l22,
                &val[blockIdx.x*(x_len+1)*(l22+1)],
                &path[blockIdx.x*(x_len+1)*(l22+1)],
                NULL);
        if(tid==0)
        {
            flag=0;
            if(dtmscore[blockIdx.x]<dtmscore2[blockIdx.x])
            {
                dtmscore[blockIdx.x]=dtmscore2[blockIdx.x];
                flag=1;
            }
        }
        if(flag)
        {
            for(int i=tid; i<l22; i=i+32)
            {
                invmapbak[blockIdx.x][i]=invmap2[blockIdx.x][0][i];
                //s[blockIdx.x*l2+i]=invmapbak[blockIdx.x][i];
            }
        }
    }

    /////////////////////////test

    //	for(int i=tid;i<l22;i=i+32)
    //		         s[blockIdx.x*l2+i]=invmapbak[blockIdx.x][i];


}

__global__ void GP_iter5(
    float x[][3],
    float y[][3],
    int x_len,
    int y_len[],
    int g1,
    int g2,
    int iteration_max,
    const int l22,
    float val[],
    char path[],
    float *s)
{


    const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    volatile __shared__  int flag;

    float  ddcc=0.4;
    if(dLnorm[blockIdx.x] <= 40) ddcc=0.1;
    if(tid==0)
        if(dtmscore2[blockIdx.x]>dtmscore[blockIdx.x]*ddcc)
            flag=1;

    if(flag)
    {
        DP_iter(x,
                &y[(blockIdx.x)*l22],
                x_len,
                y_len[blockIdx.x],
                g1,g2,iteration_max,l22,
                &val[blockIdx.x*(x_len+1)*(l22+1)],
                &path[blockIdx.x*(x_len+1)*(l22+1)],
                NULL);
        if(tid==0)
        {
            flag=0;
            if(dtmscore[blockIdx.x]<dtmscore2[blockIdx.x])
            {
                dtmscore[blockIdx.x]=dtmscore2[blockIdx.x];
                flag=1;
            }
        }
        if(flag)
        {
            for(int i=tid; i<l22; i=i+32)
            {
                invmapbak[blockIdx.x][i]=invmap2[blockIdx.x][0][i];
                //s[blockIdx.x*l2+i]=invmapbak[blockIdx.x][i];
            }
        }
    }

    /////////////////////////test

    //for(int i=tid;i<l22;i=i+32)
    //			         s[blockIdx.x*l2+i]=invmapbak[blockIdx.x][i];


}

