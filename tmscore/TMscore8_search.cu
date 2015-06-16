__device__ int score_fun8(
    float xa[][3],
    float ya[][3],
    int n_ali,
    float d,
    int i_ali[],
    float *score1,
    int score_sum_method,
    float u[3][3],
    float t[3]
)
{
    float d0=dd0[blockIdx.x];
    float score_d8=dscore_d8[blockIdx.x];
    int Lnorm=dLnorm[blockIdx.x];

    float score_sum=0, di;
    float d_tmp=d*d;
    float d02=d0*d0;
    float score_d8_cut = score_d8*score_d8;
    int i, n_cut, inc=0;

    while(1)
    {
        n_cut=0;
        score_sum=0;
        for(i=0; i<n_ali; i++)
        {
            float xt[3];
            transform(t, u, xa[i], xt);
            di = dist(xt, ya[i]);
            if(di<d_tmp)
            {
                i_ali[n_cut]=i;
                n_cut++;
            }
            if(score_sum_method==8)
            {
                if(di<=score_d8_cut)
                {
                    score_sum += 1/(1+di/d02);
                }
            }
            else
            {
                score_sum += 1/(1+di/d02);
            }
        }
        //there are not enough feasible pairs, reliefe the threshold
        if(n_cut<3 && n_ali>3)
        {
            inc++;
            float dinc=(d+inc*0.5);//------做了修改,原来值是 0.5
            d_tmp = dinc * dinc;
        }
        else
        {
            break;
        }
    }

    *score1=score_sum/Lnorm;
    //if(blockIdx.x==1)
    //	cuPrintf("Lnorm  %d  score %f \n",Lnorm,*score1);
    return n_cut;
}


__device__ float TMscore8_search(
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
    float r1[l1][3],r2[l1][3],u[3][3],t[3];
    float u0[3][3];
    float t0[3];

    const int tid=threadIdx.y*blockDim.x+threadIdx.x;

    int n_it=20;
    const int n_init_max=6;//原来6
    int L_ini[n_init_max];
    int L_ini_min=4;
    if(Lali<4) L_ini_min=Lali;
    int n_init=0, i_init;
    for(i=0; i<n_init_max-1; i++)
    {
        n_init++;
        L_ini[i]=(int) (Lali/powf(2, (float) i));
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
        //if(tid==0)
        //	cuPrintf("n_init %d Lali %d i %d\n",n_init,Lali,i);
    }

    int i_ali[l1], n_cut;
    int L_frag; //fragment length
    int iL_max; //maximum starting postion for the fragment
    i_init=threadIdx.y;
    i=0;
    if(i_init<n_init)
    //for(; i_init<n_init; i_init=i_init+blockDim.y)
    {
        L_frag=L_ini[i_init];
        iL_max=Lali-L_frag;
        i=threadIdx.x*simplify_step;
        while(1)
        {
            if(Lali-i<L_frag)
            {
                break;
            }

            //cuPrintf("======= %d %d \n",i,i_init);
            ka=0;

            for(k=0; k<L_frag; k++)
            {
                int kk=k+i;
                r1[k][0]=xtm[kk][0];
                r1[k][1]=xtm[kk][1];
                r1[k][2]=xtm[kk][2];

                r2[k][0]=ytm[kk][0];
                r2[k][1]=ytm[kk][1];
                r2[k][2]=ytm[kk][2];

                k_ali[ka]=kk;
                ka++;
            }

            Kabsch(r1, r2, L_frag, 1, &rmsd, t, u);
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
            if(i<iL_max)
            {
                i=i+simplify_step*blockDim.x;
                if(i>=iL_max) break; //i=iL_max;
            }
            else if(i>=iL_max)
                break;

        }//while(1)
    }//if(n_init)
    /**************************************************************/
    //cuPrintf("%f %d %d %d  Lali %d\n",score2,threadIdx.x,threadIdx.y,n_init,Lali);
    //if(F==1&&blockIdx.x==13)
    //cuPrintf("T: %f  %f  %f %d %d  %f  i:%d %d\n",t0[0],t0[1],t0[2],
    //threadIdx.x,threadIdx.y,score2,(i>=160? i-160:i),Lali);
    /*cuPrintf("%d %d %f  %f  %f %f\n",
    		threadIdx.x,
    		threadIdx.y,
    		t0[0],t0[1],t0[2],
    		score2);
    */
    /*************************************************************/

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

        // printf(" %f  %d\n",sscore[0],Lali);
        //dtmscore[blockIdx.x]=sscore[0];
    }

    return sscore[0];
}

__device__ void Inster_TMscore8_search(
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

                //		if(tid==0)
                //			cuPrintf("%d -> %d \n",j,i);
            }
            if(tid==0)
                sk=k;
        }
    }

    float TM=TMscore8_search(
                 xtm1[blockIdx.x],
                 ytm1[blockIdx.x],
                 sk,
                 t1[blockIdx.x],
                 u1[blockIdx.x],
                 simplify_step,
                 score_sum_method,
                 0);



    if(tid==0)
    {
        /*
        	cuPrintf("%f %f %f \n",
        		t1[blockIdx.x][0],
        		t1[blockIdx.x][1],
        		t1[blockIdx.x][2]);
        */
        //dtmscore[blockIdx.x]=TM;
        dtmscore2[blockIdx.x]=TM;
        //cuPrintf("TM == %f \n",TM);
    }
    __syncthreads();
}


__global__ void detailed_search(
    float x [][3],
    float y[][3],
    int xlen,
    int ylen[],
    int simplify_step,
    int score_sum_method,
    const int l22,
    float score[])
{
    Inster_TMscore8_search(
        x,
        &y[blockIdx.x*l22],
        xlen,
        ylen[blockIdx.x],
        invmap[blockIdx.x],
        simplify_step,
        score_sum_method,
        NULL);


}


__global__ void detailed_search1(
    float x [][3],
    float y[][3],
    int xlen,
    int ylen[],
    int simplify_step,
    int score_sum_method,
    const int l22,
    float *s)
{
    Inster_TMscore8_search(
        x,
        &y[blockIdx.x*l22],
        xlen,
        ylen[blockIdx.x],
        invmap[blockIdx.x],
        simplify_step,
        score_sum_method,
        NULL);


    const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    if(tid==0)
        dtmscore[blockIdx.x]=dtmscore2[blockIdx.x];

    for(int j=tid; j<l22; j=j+32)
    {
        invmapbak[blockIdx.x][j]=invmap[blockIdx.x][j];
        //s[blockIdx.x*l2+j]=invmap2[blockIdx.x][8][j];
    }
}


__global__ void detailed_search2(
    float x [][3],
    float y[][3],
    int xlen,
    int ylen[],
    int simplify_step,
    int score_sum_method,
    const int l22,
    float *s)
{
    Inster_TMscore8_search(
        x,
        &y[blockIdx.x*l22],
        xlen,
        ylen[blockIdx.x],
        invmap[blockIdx.x],
        simplify_step,
        score_sum_method,
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

    if(flag==1)
    {
        for(int i=tid; i<l22; i=i+32)
        {
            invmapbak[blockIdx.x][i]=invmap[blockIdx.x][i];
        }
    }

    //for(int i=tid;i<l22;i=i+32)
    //	         s[blockIdx.x*l2+i]=invmapbak[blockIdx.x][i];

}

__global__ void detailed_search3(
    float x [][3],
    float y[][3],
    int xlen,
    int ylen[],
    int simplify_step,
    int score_sum_method,
    const int l22,
    float *s)
{
    Inster_TMscore8_search(
        x,
        &y[blockIdx.x*l22],
        xlen,
        ylen[blockIdx.x],
        invmap[blockIdx.x],
        simplify_step,
        score_sum_method,
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

    if(flag==1)
    {
        for(int i=tid; i<l22; i=i+32)
        {
            invmapbak[blockIdx.x][i]=invmap[blockIdx.x][i];
        }
    }

}

__global__ void detailed_search4(
    float x [][3],
    float y[][3],
    int xlen,
    int ylen[],
    int simplify_step,
    int score_sum_method,
    const int l22,
    float *s)
{
    Inster_TMscore8_search(
        x,
        &y[blockIdx.x*l22],
        xlen,
        ylen[blockIdx.x],
        invmap[blockIdx.x],
        simplify_step,
        score_sum_method,
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

    if(flag==1)
    {

        for(int i=tid; i<l22; i=i+32)
        {

            invmapbak[blockIdx.x][i]=invmap[blockIdx.x][i];
        }
    }

}

__global__ void detailed_search5(
    float x [][3],
    float y[][3],
    int xlen,
    int ylen[],
    int simplify_step,
    int score_sum_method,
    const int l22,
    float *s)
{
    Inster_TMscore8_search(
        x,
        &y[blockIdx.x*l2],
        xlen,
        ylen[blockIdx.x],
        invmap[blockIdx.x],
        simplify_step,
        score_sum_method,
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

    if(flag==1)
    {

        for(int i=tid; i<l22; i=i+32)
        {

            invmapbak[blockIdx.x][i]=invmap[blockIdx.x][i];
        }
    }
    //for(int i=tid;i<l22;i=i+32)
    //	         s[blockIdx.x*l2+i]=invmapbak[blockIdx.x][i];

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
    Inster_TMscore8_search(
        x,
        &y[blockIdx.x*l22],
        xlen,
        ylen[blockIdx.x],
        invmapbak[blockIdx.x],
        simplify_step,
        score_sum_method,
        NULL);


}
