__device__  int find_max_frag(float x[][3],int resno[], int len)
{

    const int tid=threadIdx.y*blockIdx.x+threadIdx.x;
    unsigned short start_max,end_max;
    volatile __shared__ unsigned short start_max2;
    volatile __shared__ unsigned short end_max2;

    int r_min, fra_min=4;
    float d;
    int start;
    int Lfr_max=0, flag;

    r_min= (int) (len*1.0/3.0);
    if(r_min > fra_min) r_min=fra_min;

    int inc=0;
    float dcu0_cut=ddcu0[blockIdx.x]*ddcu0[blockIdx.x];
    float dcu_cut=dcu0_cut;

    //inc++;
    //float dinc=powf(1.1, (float) inc) * ddcu0[blockIdx.x];
    //dcu_cut= dinc*dinc;

    if(tid==0)
        while(Lfr_max < r_min)
        {
            Lfr_max=0;
            int j=1;
            start=0;
            for(int i=1; i<len; i++)
            {
                d = dist(x[i-1], x[i]);
                flag=0;
                if(dcu_cut>dcu0_cut)
                {
                    if(d<dcu_cut)
                    {
                        flag=1;
                    }
                }
		else if(resno[i] == (resno[i-1]+1)) //necessary??
		{
				if(d<dcu_cut)
				{
					flag=1;
				}
		}
                if(flag==1)
                {
                    j++;
                    if(i==(len-1))
                    {
                        if(j > Lfr_max)
                        {
                            Lfr_max=j;
                            start_max=start;
                            end_max=i;
                        }
                        j=1;
                    }
                }
                else
                {
                    if(j>Lfr_max)
                    {
                        Lfr_max=j;
                        start_max=start;
                        end_max=i-1;
                    }

                    j=1;
                    start=i;
                }
                if(Lfr_max >= r_min)
                {
                    start_max2=start_max;
                    end_max2=end_max;
                }
            }


            if(Lfr_max < r_min)
            {
                inc++;
                float dinc=powf(1.1, (float) inc) * ddcu0[blockIdx.x];
                dcu_cut= dinc*dinc;
            }

        }
    //if(tid==0)
    {
        //	cuPrintf("2-->   %d  %d   inc %d\n",start_max2,end_max2,inc);
    }
    int a=0x00000000;
    a=a|end_max2;
    a=a<<16;
    a=a|start_max2;

    return a;


}


__device__ void get_initial_fgt(
    float x[][3],
    float y[][3],
    int xresno[],
    int yresno[],
    int x_len,
    int y_len,
    int y2x2[],
    float *score)
{

    const int tid=threadIdx.y*blockIdx.x+threadIdx.x;
    int fra_min=4;
    int fra_min1=fra_min-1;

    int sd1,sd2;
    sd1=find_max_frag(x,xresno, x_len);
    sd2=find_max_frag(y,yresno,y_len);

    unsigned int s1=0x0000ffff;
    unsigned int e1=0xffff0000;
    unsigned int s2=0x0000ffff;
    unsigned int e2=0xffff0000;

    s1=s1&sd1;
    e1=e1&sd1;
    e1=e1>>16;

    s2=s2&sd2;
    e2=e2&sd2;
    e2=e2>>16;

    //if(tid==0)
    //	cuPrintf("===> %d  %d  %d   %d \n",s1,e1,s2,e2);

    int Lx = e1-s1+1;
    int Ly = e2-s2+1;
    int L_fr=(Lx<=Ly?Lx:Ly);

    /*
    	volatile __shared__ int ifr[l2];
    	if(Lx<Ly || (Lx==Ly && x_len<=y_len))
    	{
    		for(int i=tid; i<L_fr; i=i+32)
    		{
    			ifr[i]=s1+i;
    		}
    	}
    	else if(Lx>Ly || (Lx==Ly && x_len>y_len))
    	{
    		for(int i=tid; i<L_fr; i=i+32)
    		{
    			ifr[i]=s2+i;
    		}
    	}
    */

    int ifr=0;
    if(Lx<Ly || (Lx==Ly && x_len<=y_len))
    {
        ifr=s1;
    }
    else if(Lx>Ly || (Lx==Ly && x_len>y_len))
    {
        ifr=s2;
    }



    int nn1, nn2;
    int L0=(x_len<=y_len?x_len:y_len);
    /*
    if(L_fr==L0)
    {
    	nn1= (int)(L0*0.1);
    	nn2= (int)(L0*0.89);

    	int j=tid;
    	for(int i=tid+nn1; i<=nn2; i=i+32,j=j+32)
    	{
    		ifr[j]=ifr[i];
    	}
    	L_fr=nn2-nn1+1;

    }
    */

    if(L_fr==L0)
    {
        nn1= (int)(L0*0.1);
        nn2= (int)(L0*0.89);

        ifr=ifr+nn1;
        L_fr=nn2-nn1+1;

    }

    //if(tid==0)
    //	cuPrintf("L_fr %d %d %d\n",L_fr,ifr[0],ifr[L_fr-1]);

    float tmscore, tmscore_max=-1;

    int i, j, k;
    int k_best=tid;
    int min_len=0;
    if(Lx<Ly || (Lx==Ly && x_len<=y_len))
    {
        min_len=(L_fr<=y_len?L_fr: y_len);
    }
    else
    {
        min_len=(x_len<=L_fr?x_len:L_fr);
    }


    int min_ali= (int) (min_len/2.5);
    if(min_ali<=fra_min1)  min_ali=fra_min1;

    if(Lx<Ly || (Lx==Ly && x_len<=y_len))
    {
        nn1 = -y_len+min_ali;
        nn2 = L_fr-min_ali;
    }
    else
    {
        nn1 = -L_fr+min_ali;
        nn2 = x_len-min_ali;
    }

    //if(tid==0)
    //	cuPrintf("%d  %d  %d \n",L_fr,nn1,nn2);


    int  y2x_[l2];
    for(k=nn1+tid; k<=nn2; k=k+32)
    {
        for(j=0; j<y_len; j++)
            y2x_[j]=-1;

        if(Lx<Ly || (Lx==Ly && x_len<=y_len))
        {
            for(j=0; j<y_len; j++)
            {
                i=j+k;
                if(i>=0 && i<L_fr)
                {
                    y2x_[j]=ifr+i;
                }
            }
        }
        else
        {
            for(j=0; j<L_fr; j++)
            {
                i=j+k;
                if(i>=0 && i<x_len)
                {
                    y2x_[ifr+j]=i;
                }
            }

        }
        /*
        	if(k==-42)
                    {
                            for(int m=0;m<y_len;m++)
                                   	if(m<L_fr)
                                   	cuPrintf("%d %d\n",ifr[m],y2x_[m]);
                           		else
                                   	cuPrintf("   %d\n",y2x_[m]);
                    }
        */
        tmscore=get_score_fast(x, y, x_len, y_len, y2x_);
        if(tmscore>=tmscore_max)
        {
            tmscore_max=tmscore;
            k_best=k;
        }
    }


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
    }

    k=sscore_i[0];

    //if(tid==0)
    //{
    //if(blockIdx.x<10)
    //	printf("0%d (%d %d) %d  %f\n",blockIdx.x,nn1,nn2,sscore_i[0],sscore[0]);
    //else
    //	printf("%d (%d %d) %d  %f\n",blockIdx.x,nn1,nn2,sscore_i[0],sscore[0]);
    //cuPrintf("1 k %d tm %f == 2 k %d  tm %f\n",sscore_i[0],sscore[0],sscore_i[1],sscore[1]);
    //}

    for(j=tid; j<y_len; j=j+32)
        y2x2[j]=-1;


    if(Lx<Ly || (Lx==Ly && x_len<=y_len))
    {
        for(j=tid; j<y_len; j=j+32)
        {
            i=j+k;
            if(i>=0 && i<L_fr)
            {
                y2x2[j]=ifr+i;
            }
        }
    }
    else
    {
        for(j=tid; j<L_fr; j=j+32)
        {
            i=j+k;
            if(i>=0 && i<x_len)
            {
                y2x2[ifr+j]=i;
            }
        }
    }
}
__global__ void get_initial_fgt2(
    float x[][3],
    float y[][3],
    int xresno[],
    int yresno[],
    int x_len,
    int y_len[],
    const int l22,
    float *s)
{
    //cuPrintf("== %d \n",y_len[blockIdx.x]);
    //if(blockIdx.x==7||blockIdx.x==8)
    {
        get_initial_fgt(
            x,
            &y[blockIdx.x*l22],
	    &xresno[blockIdx.x*l22],
    	    &yresno[blockIdx.x*l22],
            x_len,
            y_len[blockIdx.x],
            invmap[blockIdx.x],
            NULL);

        /*
        	const int tid=threadIdx.y*blockIdx.x+threadIdx.x;
        	int i=0;
        	for(i=tid;i<l2;i=i+32)
        	{
        		s[blockIdx.x*l2+i]=invmap[blockIdx.x][i];
        	}
        */
    }
}
