__device__ void NWDP_TM2(
    int secx[],
    int secy[],
    int len1,
    int len2,
    float gap_open,
    int j2i[],
    const int l,
    float val[],
    char path[]
)
{

    int i, j;
    float h, v, d;

    const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    for(i=tid; i<=len1; i+=32)
    {
        val[i*(l+1)+0]=0;
        path[i*(l+1)+0]=0;
    }

    for(j=tid; j<=len2; j+=32)
    {
        val[j]=0;
        path[j]=0;
        j2i[j]=-1;
    }

    int nv=1;
    i=tid+1;
    j=1;

    while(i<=len1)
    {
        if(i<=nv)
        {
            if(j<=len2)
            {

                if(secx[i-1]==secy[j-1])
                {
                    d=val[(i-1)*(l+1)+j-1]+ 1.0;
                }
                else
                {
                    d=val[(i-1)*(l+1)+j-1];
                }


                h=val[(i-1)*(l+1)+j];
                if(path[(i-1)*(l+1)+j])
                    h += gap_open;

                v=val[i*(l+1)+j-1];
                if(path[i*(l+1)+j-1])
                    v += gap_open;


                if(d>=h && d>=v)
                {
                    path[i*(l+1)+j]=1;
                    val[i*(l+1)+j]=d;
                }
                else
                {
                    path[i*(l+1)+j]=0;
                    if(v>=h)
                        val[i*(l+1)+j]=v;
                    else
                        val[i*(l+1)+j]=h;
                }
                j++;
            }
        }
        nv++;
        if(j>len2)
        {
            i+=32;
            j=1;
        }
    }

    if(tid==0)
    {
        i=len1;
        j=len2;
        while(i>0 && j>0)
        {
            if(path[i*(l+1)+j])
            {
                j2i[j-1]=i-1;
                i--;
                j--;
            }
            else
            {
                h=val[(i-1)*(l+1)+j];
                if(path[(i-1)*(l+1)+j]) h +=gap_open;

                v=val[i*(l+1)+j-1];
                if(path[i*(l+1)+j-1]) v +=gap_open;

                if(v>=h)
                    j--;
                else
                    i--;
            }
        }
    }
}

__device__ void smooth(int *sec, int len)
{
    int i, j;
    //smooth single  --x-- => -----
    for(i=2; i<len-2; i++)
    {
        if(sec[i]==2 || sec[i]==4)
        {
            j=sec[i];
            if(sec[i-2] != j)
            {
                if(sec[i-1] != j)
                {
                    if(sec[i+1] != j)
                    {
                        if(sec[i+2] != j)
                        {
                            sec[i]=1;
                        }
                    }
                }
            }
        }
    }

    //   smooth float
    //   --xx-- => ------

    for(i=0; i<len-5; i++)
    {
        //helix
        if(sec[i] != 2)
        {
            if(sec[i+1] != 2)
            {
                if(sec[i+2] == 2)
                {
                    if(sec[i+3] == 2)
                    {
                        if(sec[i+4] != 2)
                        {
                            if(sec[i+5] != 2)
                            {
                                sec[i+2]=1;
                                sec[i+3]=1;
                            }
                        }
                    }
                }
            }
        }

        //beta
        if(sec[i] != 4)
        {
            if(sec[i+1] != 4)
            {
                if(sec[i+2] ==4)
                {
                    if(sec[i+3] == 4)
                    {
                        if(sec[i+4] != 4)
                        {
                            if(sec[i+5] != 4)
                            {
                                sec[i+2]=1;
                                sec[i+3]=1;
                            }
                        }
                    }
                }
            }
        }
    }

    //smooth connect
    for(i=0; i<len-2; i++)
    {
        if(sec[i] == 2)
        {
            if(sec[i+1] != 2)
            {
                if(sec[i+2] == 2)
                {
                    sec[i+1]=2;
                }
            }
        }
        else if(sec[i] == 4)
        {
            if(sec[i+1] != 4)
            {
                if(sec[i+2] == 4)
                {
                    sec[i+1]=4;
                }
            }
        }
    }

}

__device__ int sec_str(float dis13, float dis14,
                       float dis15, float dis24,
                       float dis25, float dis35)
{
    int s=1;

    float delta=2.1;
    if(fabs(dis15-6.37)<delta)
    {
        if(fabs(dis14-5.18)<delta)
        {
            if(fabs(dis25-5.18)<delta)
            {
                if(fabs(dis13-5.45)<delta)
                {
                    if(fabs(dis24-5.45)<delta)
                    {
                        if(fabs(dis35-5.45)<delta)
                        {
                            s=2;
                            return s;
                        }
                    }
                }
            }
        }
    }

    delta=1.42;
    if(fabs(dis15-13)<delta)
    {
        if(fabs(dis14-10.4)<delta)
        {
            if(fabs(dis25-10.4)<delta)
            {
                if(fabs(dis13-6.1)<delta)
                {
                    if(fabs(dis24-6.1)<delta)
                    {
                        if(fabs(dis35-6.1)<delta)
                        {
                            s=4; //strand
                            return s;
                        }
                    }
                }
            }
        }
    }

    if(dis15 < 8)
    {
        s=3; //turn
    }


    return s;
}


__device__ void make_sec(float x[][3], int len, int sec[])
{

    const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    for(int i=tid; i<len; i=i+32)
    {
        sec[i]=-1;
    }


    int j1, j2, j3, j4, j5;
    float d13, d14, d15, d24, d25, d35;
    for(int i=tid; i<len; i+=32)
    {
        sec[i]=1;
        j1=i-2;
        j2=i-1;
        j3=i;
        j4=i+1;
        j5=i+2;

        if(j1>=0 && j5<len)
        {
            d13=sqrt(dist(x[j1], x[j3]));
            d14=sqrt(dist(x[j1], x[j4]));
            d15=sqrt(dist(x[j1], x[j5]));
            d24=sqrt(dist(x[j2], x[j4]));
            d25=sqrt(dist(x[j2], x[j5]));
            d35=sqrt(dist(x[j3], x[j5]));
            sec[i]=sec_str(d13, d14, d15, d24, d25, d35);
        }

    }
    if(tid==0)
        smooth(sec, len);
}

__device__  void get_initial_ss(
    float x[][3],
    float y[][3],
    int secx[],
    int secy[],
    int x_len,
    int y_len,
    int *y2x,
    const int l,
    float val[],
    char path[])
{

    make_sec(x, x_len, secx);
    make_sec(y, y_len, secy);

    float gap_open=-1.0;
    NWDP_TM2(secx,
             secy,
             x_len,
             y_len,
             gap_open,
             y2x,
             l,
             val,
             path);

}

__global__ void get_initial_ss2(
    float x[][3],
    float y[][3],
    int x_len,
    int y_len[],
    const int l22,
    float val[],
    char path[],
    float *s)
{

    get_initial_ss(	x,
                    &y[blockIdx.x*l22],
                    secx[blockIdx.x],
                    secy[blockIdx.x],
                    x_len,
                    y_len[blockIdx.x],
                    invmap[blockIdx.x],l22,
                    &val[blockIdx.x*(x_len+1)*(l22+1)],
                    &path[blockIdx.x*(x_len+1)*(l22+1)]);


    /*
    	const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    	for(int i=tid;i<l2;i=i+32)
    	{
    		s[blockIdx.x*l2+i]=invmap[blockIdx.x][i];
    	}
    */
}


