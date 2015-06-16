
__device__ float get_initial_local(
    float x[][3],
    float y[][3],
    int x_len,
    int y_len,
    int *y2x,
    const int l,
    float val[],
    char path[])
{
    float GL, rmsd;
    float d01=dd0[blockIdx.x]+1.5;
    if(d01 < dD0_MIN[blockIdx.x]) d01=dD0_MIN[blockIdx.x];
    float d02=d01*d01;

    float GLmax=0;
    int n_frag=20;
    int ns=20;

    int aL=(x_len<=y_len?x_len:y_len);
    if(aL>250)
    {
        n_frag=50;
    }
    else if(aL>200)
    {
        n_frag=40;
    }
    else if(aL>150)
    {
        n_frag=30;
    }
    else
    {
        n_frag=20;
    }

    int smallest=aL/3;

    if(n_frag>smallest) n_frag=smallest;
    if(ns>smallest) ns=smallest;

    int m1=x_len-n_frag-ns;
    int m2=y_len-n_frag-ns;
    int i,j,k;
    const int tid=threadIdx.y*blockDim.x+threadIdx.x;

    float r1[32][3],r2[32][3],u[3][3],t[3];
    __shared__ float t0[32][3],u0[32][3][3];
//	volatile __shared__ int sinvmap[l2];

    volatile __shared__ int sk;
    volatile __shared__ float sGLmax[32];
    sGLmax[tid]=-1;
    i=ns-1;
    for(; i<m1; i=i+n_frag)
    {

        j=ns-1+tid*n_frag;
        if(j<m2)
        {
            for(k=0; k<n_frag; k++)
            {
                r1[k][0]=x[k+i][0];
                r1[k][1]=x[k+i][1];
                r1[k][2]=x[k+i][2];

                r2[k][0]=y[k+j][0];
                r2[k][1]=y[k+j][1];
                r2[k][2]=y[k+j][2];
            }
            Kabsch(r1, r2, n_frag, 1, &rmsd, t, u);
            for(k=0; k<3; k++)
            {
                t0[tid][k]=t[k];
                u0[tid][k][0]=u[k][0];
                u0[tid][k][1]=u[k][1];
                u0[tid][k][2]=u[k][2];
            }

            //printf(" %d %f %f %f\n",tid,t0[tid][0],t0[tid][1],t0[tid][2]);
        }
        /*

        		k=0;
        		for(j=ns-1; j<m2; j=j+n_frag)
        		{
        			float gap_open=0.0;
        			DNW(x, y,
        				x_len,
        				y_len,
                        		t0[k],
                        		u0[k],
                        		d02,
                        		gap_open,
                        		invmap[blockIdx.x],
                       			val, path);
        			k++;
        			if(tid<k)
        			{
        				GL=get_score_fast(
        					x,
        					y,
        				 	x_len,
        					y_len,
        					invmap[blockIdx.x]);
        					//printf("%d %f \n",tid,GL);
        				if(GL>GLmax)
        				{
        					GLmax=GL;

        					for(int k1=tid;k1<l2;k1=k1+32)
        					sinvmap[k1]=invmap[blockIdx.x][k1];

        				}
        			}
        		}

        */

        k=0;
        for(j=ns-1; j<m2; j=j+n_frag)
        {
            float gap_open=0.0;
            DNW(x, y,
                x_len,
                y_len,
                t0[k],
                u0[k],
                d02,
                gap_open,
                invmap2[blockIdx.x][k],
		l,
                val, path);
            k++;
        }

        if(tid<k)
        {
            GL=get_score_fast(
                   x,
                   y,
                   x_len,
                   y_len,
                   invmap2[blockIdx.x][tid]);
            //printf("%f\n",GL);
            if(GL>sGLmax[tid])
            {
                sGLmax[tid]=GL;
            }
        }

        if( tid==0)
        {
            int i;
            GLmax=0;
            for(i=0; i<8; i++)
                if(GLmax<sGLmax[i])
                {
                    GLmax=sGLmax[i];
                    sk=i;
                }
        }

        if(sGLmax[9]<sGLmax[sk])
        {
            for(k=tid; k<l; k=k+32)
                invmap[blockIdx.x][k]=invmap2[blockIdx.x][sk][k];
            if(tid==0)
                sGLmax[9]=sGLmax[sk];
        }
    }
    GLmax=sGLmax[9];
    //if(threadIdx.x==0&&threadIdx.y==0)
    //	printf("%f \n",GLmax);
    return GLmax;
}

__global__ void get_initial_local2(
    float x[][3],
    float y[][3],
    int x_len,
    int y_len[],
    const int l22,
    float val[],
    char path[],
    float *s)
{
    //float GL=
    get_initial_local(
        x,
        &y[blockIdx.x*l22],
        x_len,
        y_len[blockIdx.x],
        invmap[blockIdx.x],l22,
        &val[blockIdx.x*(x_len+1)*(l22+1)],
        &path[blockIdx.x*(x_len+1)*(l22+1)]);
    /*
    	const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    	//if(threadIdx.x==0&&threadIdx.y==0)
    	//	s[blockIdx.x]=GL;
    	for(int k=tid;k<l2;k=k+32)
    		s[blockIdx.x*l2+k]=invmap[blockIdx.x][k];

    */
}
