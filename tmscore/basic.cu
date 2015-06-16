__device__ float dist(float x[3], float y[3])
{
    float d1=x[0]-y[0];
    float d2=x[1]-y[1];
    float d3=x[2]-y[2];

    return (d1*d1 + d2*d2 + d3*d3);
}

__device__ float dot(float a[], float b[])
{
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

__device__ void transform(float t[3], float u[3][3], float *x, float *x1)
{
    x1[0]=t[0]+dot(&u[0][0], x);
    x1[1]=t[1]+dot(&u[1][0], x);
    x1[2]=t[2]+dot(&u[2][0], x);
}

__device__ void do_rotation(float x[][3], float x1[][3], int len, float t[3], float u[3][3])
{
    int i=0;
    for(; i<len; i++)
    {
        transform(t,u,&x[i][0],&x1[i][0]);
    }
}

char* get50(int a,char *c)
{
    c[0]=c[1]=c[2]=c[3]=c[4]='0';
    c[5]='\0';
    if(a<10)	{
        c[4]='\0';
        return c;
    }
    if(a<100)	{
        c[3]='\0';
        return c;
    }
    if(a<1000)	{
        c[2]='\0';
        return c;
    }
    if(a<10000)	{
        c[1]='\0';
        return c;
    }
    if(a<100000) {
        c[0]='\0';
        return c;
    }

    return c;
}
