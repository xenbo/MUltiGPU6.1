
__device__ float TMscore8_search3(
    float xtm[][3],
    float ytm[][3],
    int Lali,
    float t1[3],
    float u1[3][3],
    int simplify_step,
    int score_sum_method,
    int F
)
{
    int i, m;
    float d0_search=dd0_search[blockIdx.x];
    float  score=-1.0,score2=-1.0, rmsd;
    int k_ali[l1], ka, k;
    float d;
    float u[3][3],t[3];
    float u0[3][3];
    float t0[3];
	int i_ali[l1], n_cut;
    const int tid=threadIdx.y*blockDim.x+threadIdx.x;

    const int n_it=20;
    const int n_init_max=6;
    int L_ini[n_init_max];
    int L_ini_min=4;
    if(Lali<4) L_ini_min=Lali;
    int n_init=0;
    for(i=0; i<n_init_max-1; i++)
    {
        n_init++;
        L_ini[i]=(int) (Lali/__powf(2.0, (float) i));
          
        if(L_ini[i]<=L_ini_min)
        {
            L_ini[i]=L_ini_min;
            break;
        }
    }
    if(i==n_init_max-1)
    {
        n_init++;
        L_ini[i]=L_ini_min;
    }


    const int nu[]= {1,3,5,7,9,7};
    const int hh=(int)__powf(tid,(float)0.5);

    if(hh<n_init)
    {
        //printf("%d  %d  %d\n",tid,hh,L_ini[hh]);

        const int s=tid-hh*hh;
        for(int jid=s; jid<=Lali-L_ini[hh]; jid=jid+nu[hh])
        {

            //printf("%d-%d \n",jid,jid+L_ini[hh]);

            k=0;
            float r1[l1][3],r2[l1][3];
            for(int kk=jid; kk<jid+L_ini[hh]; kk++)
            {

                r1[k][0]=xtm[kk][0];
                r1[k][1]=xtm[kk][1];
                r1[k][2]=xtm[kk][2];

                r2[k][0]=ytm[kk][0];
                r2[k][1]=ytm[kk][1];
                r2[k][2]=ytm[kk][2];
                k++;
            }

            Kabsch(r1, r2, k, 1, &rmsd, t, u);
            d=d0_search-1;
            n_cut=score_fun8(xtm, ytm, Lali, d, i_ali, &score, score_sum_method,u,t);

            if(score>score2)
            {
                score2=score;
                for(k=0; k<3; k++)
                {
                    t0[k]=t[k];
                    u0[k][0]=u[k][0];
                    u0[k][1]=u[k][1];
                    u0[k][2]=u[k][2];
                }
            }

            d=d0_search+1;
            int it=0;
            for(; it<n_it; it++)
            {
                ka=0;
                for(k=0; k<n_cut; k++)
                {
                    m=i_ali[k];
                    r1[k][0]=xtm[m][0];
                    r1[k][1]=xtm[m][1];
                    r1[k][2]=xtm[m][2];

                    r2[k][0]=ytm[m][0];
                    r2[k][1]=ytm[m][1];
                    r2[k][2]=ytm[m][2];

                    k_ali[ka]=m;
                    ka++;
                }
                Kabsch(r1, r2, n_cut, 1, &rmsd, t, u);

                n_cut=score_fun8(xtm, ytm, Lali, d, i_ali, &score, score_sum_method,u,t);
                if(score>score2)
                {
                    score2=score;
                    for(k=0; k<3; k++)
                    {
                        t0[k]=t[k];
                        u0[k][0]=u[k][0];
                        u0[k][1]=u[k][1];
                        u0[k][2]=u[k][2];
                    }
                }

                if(n_cut==ka)
                {
                    for(k=0; k<n_cut; k++)
                    {
                        if(i_ali[k]!=k_ali[k])
                        {
                            break;
                        }
                    }
                    if(k==n_cut)
                    {
                        break; //stop iteration
                    }
                }
            } //for iteration
        }
    }
    __syncthreads();
    volatile __shared__ float sscore[32];
    volatile __shared__ int sscore_i[32];
    sscore_i[tid]=tid;
    sscore[tid]=score2;
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
    }
    if(tid==sscore_i[0])
    {
        int k= 0;
        for(k=0; k<3; k++)
        {   t1[k]=t0[k];
            u1[k][0]=u0[k][0];
            u1[k][1]=u0[k][1];
            u1[k][2]=u0[k][2];

        }
        //cuPrintf("%f %f %f \n",	t1[0],t1[1],t1[2]);

        //cuPrintf(" %f  %d\n",sscore[0],Lali);
        //dtmscore[blockIdx.x]=sscore[0];
    }

    return sscore[0];

}

__device__ void Inster_TMscore8_search3(
    float x[][3],
    float y[][3],
    int xlen,
    int ylen,
    int map[],
    int simplify_step,
    int score_sum_method,
    float *s
)
{
    int j=0;
    int k=0;
    const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    volatile __shared__ int sk;
    if(tid<3)
    {
        for(j=0; j<ylen; j++)
        {
            int i=map[j];
            if(i>=0&&i<xlen)
            {
                xtm1[blockIdx.x][k][tid]=x[i][tid];
                ytm1[blockIdx.x][k][tid]=y[j][tid];
                k++;
            }
            if(tid==0)
                sk=k;
        }
    }

    float TM=TMscore8_search3(
                 xtm1[blockIdx.x],
                 ytm1[blockIdx.x],
                 sk,
                 t1[blockIdx.x],
                 u1[blockIdx.x],
                 simplify_step,
                 score_sum_method,
                 0);

	if(tid==0)
        dtmscore2[blockIdx.x]=TM;
   

}


__global__ void detailed_search6(
    float x [][3],
    float y[][3],
    int xlen,
    int ylen[],
    int simplify_step,
    int score_sum_method,
    const int l22,
    float score[])
{
    Inster_TMscore8_search3(
        x,
        &y[blockIdx.x*l22],
        xlen,
        ylen[blockIdx.x],
        invmapbak[blockIdx.x],
        simplify_step,
        score_sum_method,
        NULL);

}
