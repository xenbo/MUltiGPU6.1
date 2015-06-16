__device__ void DNW(
    float  x[][3],
    float  y[][3],
    int len1,
    int len2,
    float t[3],
    float u[3][3],
    float d02,
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

    float xx[3], dij;
    int nv=1;
    i=tid+1;
    j=1;
    while(i<=len1)
    {
        if(i<=nv)
        {
            if(j<=len2)
            {
                transform(t, u, &x[i-1][0], xx);
                dij=dist(xx, &y[j-1][0]);
                d=val[(i-1)*(l+1)+j-1] +  1.0/(1+dij/d02);

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

