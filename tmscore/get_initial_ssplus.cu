__device__ void NWDP_TM3(
    int len1,
    int len2,
    float gap_open,
    int j2i[],
    float score[],
    const int l,
    float val[],
    char path[])
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
                d=val[(i-1)*(l+1)+j-1]+score[i*(l+1)+j];

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

__device__ void score_matrix_rmsd_sec(
    float x[][3],
    float y[][3],
    int x_len,
    int y_len,
    int secx[],
    int secy[],
    const int l,
    int *y2x,
    float score[])
{
    const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    __shared__ float t[3], u[3][3];
    float rmsd, dij;
    float d01=dd0[blockIdx.x]+1.5;
    if(d01 < dD0_MIN[blockIdx.x]) d01=dD0_MIN[blockIdx.x];
    float d02=d01*d01;

    float xx[3];
    if(tid==0)
    {
        int i, k=0;
        float r1[l1][3],r2[l1][3];
        for(int j=0; j<y_len; j++)
        {
            i=y2x[j];
            if(i>=0&&i<x_len)
            {
                r1[k][0]=x[i][0];
                r1[k][1]=x[i][1];
                r1[k][2]=x[i][2];

                r2[k][0]=y[j][0];
                r2[k][1]=y[j][1];
                r2[k][2]=y[j][2];

                k++;
            }
        }
        Kabsch(r1, r2, k, 1, &rmsd, t, u);
    }
    for(int ii=threadIdx.x; ii<x_len; ii+=blockDim.x)
    {
        transform(t, u, &x[ii][0], xx);
        for(int jj=threadIdx.y; jj<y_len; jj+=blockDim.y)
        {
            dij=dist(xx, &y[jj][0]);
            if(secx[ii]==secy[jj])
            {
                score[(ii+1)*(l+1)+jj+1] = 1.0/(1+dij/d02) + 0.5;
            }
            else
            {
                score[(ii+1)*(l+1)+jj+1] = 1.0/(1+dij/d02);
            }
        }
    }
}


__device__ void get_initial_ssplus(
    float x[][3],
    float y[][3],
    int x_len,
    int y_len,
    int secx[],
    int secy[],
    int *y2xb,
    int *y2x,
    const int l,
    float score[],
    float val[],
    char path[])
{

    score_matrix_rmsd_sec(x, y, x_len, y_len,secx,secy,l,y2xb,score);
    float gap_open=-1.0;
    NWDP_TM3(x_len, y_len, gap_open, y2x,score,l,val,path);
}

__global__ void get_initial_ssplus2(
    float x[][3],
    float y[][3],
    int x_len,
    int y_len[],
    const int l22,
    float score[],
    float val[],
    char path[],
    float *s)
{

    get_initial_ssplus( x,
                        &y[blockIdx.x*l22],
                        x_len,
                        y_len[blockIdx.x],
                        secx[blockIdx.x],
                        secy[blockIdx.x],
                        invmapbak[blockIdx.x],
                        invmap[blockIdx.x],l22,
                        &score[blockIdx.x*(x_len+1)*(l22+1)],
                        &val[blockIdx.x*(x_len+1)*(l22+1)],
                        &path[blockIdx.x*(x_len+1)*(l22+1)]
                      );


    /*
    	const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    	for(int i=tid;i<l2;i=i+32)
    	{
    		s[blockIdx.x*l2+i]=invmap[blockIdx.x][i];
    	}
    */
}

