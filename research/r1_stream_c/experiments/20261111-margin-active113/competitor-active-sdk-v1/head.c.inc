/* Five exact rows and ONLY113 selected OTHER rows; original par4 operations. */
static uint32_t affine_qhead(const r1active113_model *m,const double *x,double *y) {
    for(unsigned k=0;k<5;k++) {
        const float *row=m->precise_head+(size_t)k*140u;double sum=0.0;
        for(unsigned i=0;i<140;i++)sum+=x[i]*(double)row[i];
        y[k]=sum+(double)m->precise_biases[k];
    }
    double maximum=-INFINITY;uint32_t winner=UINT32_MAX;
    for(unsigned o=0;o<R1ACTIVE113_OTHER_COUNT;o++) {
        const int8_t *row=m->qhead+(size_t)o*140u;
        double s0=0.0,s1=0.0,s2=0.0,s3=0.0;
        for(unsigned i=0;i<140;i+=4) {
            s0+=x[i]*(double)row[i];s1+=x[i+1]*(double)row[i+1];
            s2+=x[i+2]*(double)row[i+2];s3+=x[i+3]*(double)row[i+3];
        }
        double sum=(s0+s1)+(s2+s3);sum*=m->scales[o];
        double value=sum+(double)m->biases[o];
        if(!isfinite(value)){y[5]=value;return m->other_ids[o];}
        /* Frozen selected IDs ascend, so strict> retains lowest-ID ties. */
        if(value>maximum){maximum=value;winner=m->other_ids[o];}
    }
    y[5]=maximum;return winner;
}
