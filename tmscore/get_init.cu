__device__ float get_score_fast(float x[][3],float y[][3],int x_len, int y_len, int map[])
{
    float rms, tmscore, tmscore1, tmscore2,r1[l1][3],r2[l1][3],xtm[l1][3],ytm[l1][3],u[3][3],t[3];
    int i, j, k;

    k=0;
    for(j=0; j<y_len; j++)
    {
        i=map[j];
        if(i>=0&&i<x_len)
        {
            /*
            r1[k][0]=x[i][0];
            r1[k][1]=x[i][1];
            r1[k][2]=x[i][2];

            r2[k][0]=y[j][0];
            r2[k][1]=y[j][1];
            r2[k][2]=y[j][2];
            */

            xtm[k][0]=x[i][0];
            xtm[k][1]=x[i][1];
            xtm[k][2]=x[i][2];

            ytm[k][0]=y[j][0];
            ytm[k][1]=y[j][1];
            ytm[k][2]=y[j][2];

            k++;
        }

    }
    Kabsch(xtm, ytm, k, 1, &rms, t, u);
    // cuPrintf("%f %f %f %f %f %f \n",r1[0][0],r1[0][1],r1[0][2],u[0][0],u[0][1],u[0][2]);
    float di;
    float dis[l1];
    float d0_search=dd0_search[blockIdx.x];
    float d00=d0_search;
    float d002=d00*d00;
    float d0=dd0[blockIdx.x];
    float d02=d0*d0;

    int n_ali=k;
    float xrot[3];
    tmscore=0;
    for(k=0; k<n_ali; k++)
    {
        transform(t, u, &xtm[k][0], xrot);
        di=dist(xrot, &ytm[k][0]);
        dis[k]=di;
        tmscore +=  1/(1+di/d02);
    }

    //second iteration
    float d002t=d002;
    while(1)
    {
        j=0;
        for(k=0; k<n_ali; k++)
        {
            if(dis[k]<=d002t)
            {
                r1[j][0]=xtm[k][0];
                r1[j][1]=xtm[k][1];
                r1[j][2]=xtm[k][2];

                r2[j][0]=ytm[k][0];
                r2[j][1]=ytm[k][1];
                r2[j][2]=ytm[k][2];

                j++;
            }
        }

        if(j<3 && n_ali>3)
        {
            d002t += 0.5;//------做了修改(默认0.5)
        }
        else
        {
            break;
        }
    }

    if(n_ali!=j)
    {
        Kabsch(r1, r2, j, 1, &rms, t, u);
        tmscore1=0;
        for(k=0; k<n_ali; k++)
        {
            transform(t, u, &xtm[k][0], xrot);
            di=dist(xrot, &ytm[k][0]);
            dis[k]=di;
            tmscore1 += 1/(1+di/d02);
        }

        d002t=d002+1;

        while(1)
        {
            j=0;
            for(k=0; k<n_ali; k++)
            {
                if(dis[k]<=d002t)
                {
                    r1[j][0]=xtm[k][0];
                    r1[j][1]=xtm[k][1];
                    r1[j][2]=xtm[k][2];

                    r2[j][0]=ytm[k][0];
                    r2[j][1]=ytm[k][1];
                    r2[j][2]=ytm[k][2];

                    j++;
                }
            }

            if(j<3 && n_ali>3)
            {
                d002t += 0.5;//------ 做了修改（默认0.5）
            }
            else
            {
                break;
            }
        }

        Kabsch(r1, r2, j, 1, &rms, t, u);
        tmscore2=0;
        for(k=0; k<n_ali; k++)
        {
            transform(t, u, &xtm[k][0], xrot);
            di=dist(xrot, &ytm[k][0]);
            tmscore2 += 1/(1+di/d02);
        }
    }
    else
    {
        tmscore1=tmscore;
        tmscore2=tmscore;
    }

    if(tmscore1>=tmscore) tmscore=tmscore1;
    if(tmscore2>=tmscore) tmscore=tmscore2;

    return tmscore;
}


__device__ void get_initial(
    float x[][3],
    float y[][3],
    int x_len,
    int y_len,
    int *y2x2,
    float *s)
{
    int min_len=(x_len<=y_len? x_len:y_len);
    int min_ali= min_len/2;
    if(min_ali<5) min_ali=5;

    int nn1= min_ali-y_len;
    int nn2=x_len-min_ali;

    int  y2x[l2];
    int i, j, k, k_best;
    float tmscore=-1, tmscore_max=-1;

    const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    k_best=nn1+tid;
    for(k=nn1+tid; k<=nn2; k=k+blockDim.x)
    {
        for(j=0; j<y_len; j++)
        {
            i=j+k;
            if(i>=0 && i<x_len)
            {
                y2x[j]=i;
            }
            else
            {
                y2x[j]=-1;
            }
        }


        tmscore=get_score_fast(x, y, x_len, y_len, y2x);
        if(tmscore>=tmscore_max)
        {
            tmscore_max=tmscore;
            k_best=k;
        }
        //s[tid]=tmscore;
    }


    //合并归约
    volatile __shared__ float sscore[32];
    volatile __shared__  int  sscore_i[32];
    sscore_i[tid]=k_best;
    sscore[tid]=tmscore_max;

    if(tid<16)
    {
        if(sscore[tid]<sscore[tid+16])
        {
            sscore[tid]=sscore[tid+16];
            sscore_i[tid]=sscore_i[tid+16];
        }
    }

    if(tid<8)
    {
        if(sscore[tid]<sscore[tid+8])
        {
            sscore[tid]=sscore[tid+8];
            sscore_i[tid]=sscore_i[tid+8];
        }
    }
    if(tid<4)
    {
        if(sscore[tid]<sscore[tid+4])
        {
            sscore[tid]=sscore[tid+4];
            sscore_i[tid]=sscore_i[tid+4];
        }
    }
    if(tid<2)
    {
        if(sscore[tid]<sscore[tid+2])
        {
            sscore[tid]=sscore[tid+2];
            sscore_i[tid]=sscore_i[tid+2];
        }
    }
    if(tid<1)
    {
        if(sscore[tid]<sscore[tid+1])
        {
            sscore[tid]=sscore[tid+1];
            sscore_i[tid]=sscore_i[tid+1];
        }
        //printf("sscore %f k_best %d \n",sscore[0],sscore_i[0]);
        //*s=sscore[0];
    }

    k=sscore_i[0];
    for(j=tid; j<y_len; j=j+blockDim.x)
    {
        i=j+k;
        if(i>=0 && i<x_len)
        {
            y2x2[j]=i;
        }
        else
        {
            y2x2[j]=-1;
        }
    }
}
__global__ void get_initial2( float x[][3],
                              float y[][3],
                              int x_len,
                              int y_len[],
                              const int l22,
                              float *s
                            )
{
    //cuPrintf("== %d \n",y_len[blockIdx.x]);
    //if(blockIdx.x==14)
    get_initial(
        x,
        &y[blockIdx.x*l22],
        x_len,
        y_len[blockIdx.x],
        invmap[blockIdx.x],
        NULL);

}
