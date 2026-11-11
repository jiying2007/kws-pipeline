/* Strict ACTIVE113 package loaders; no model arithmetic in these functions. */
static int valid(const r1active113_model *m) {
    return m && m->magic==R1ACTIVE113_MAGIC && m->payload && m->weights && m->other_ids
        && m->qhead && m->scales && m->biases && m->precise_head && m->precise_biases;
}
static uint32_t le32(const unsigned char *p) {
    return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}
static r1active113_status check_package(const unsigned char *p,size_t n,unsigned order,const char hash[65]) {
    if(n!=R1ACTIVE113_PAYLOAD_BYTES)return R1ACTIVE113_SIZE;
    if(order!=R1ACTIVE113_PAYLOAD_LITTLE_ENDIAN || !r1active113_environment_ok())return R1ACTIVE113_ENVIRONMENT;
    for(unsigned i=0;i<64;i++)
        if(!((hash[i]>='0' && hash[i]<='9') || (hash[i]>='a' && hash[i]<='f')))return R1ACTIVE113_IDENTITY;
    if(hash[64]!='\0')return R1ACTIVE113_IDENTITY;
    char actual[65];r1a113_sha256_memory_hex(p,n,actual);
    if(memcmp(actual,hash,65))return R1ACTIVE113_IDENTITY;
    if(memcmp(p,R1ACTIVE113_EXPECTED_HEADER,R1ACTIVE113_HEADER_BYTES))return R1ACTIVE113_IDENTITY;
    for(unsigned i=0;i<5;i++)
        if(le32(p+R1ACTIVE113_IDS_OFFSET+i*4u)!=R1ACTIVE113_PRECISE_IDS[i])return R1ACTIVE113_IDENTITY;
    for(unsigned i=0;i<R1ACTIVE113_OTHER_COUNT;i++)
        if(le32(p+R1ACTIVE113_IDS_OFFSET+(i+5u)*4u)!=R1ACTIVE113_OTHER_IDS[i])return R1ACTIVE113_IDENTITY;
    for(size_t i=R1ACTIVE113_QUANT_OFFSET+R1ACTIVE113_OTHER_COUNT*140u;i<R1ACTIVE113_SCALE_OFFSET;i++)
        if(p[i])return R1ACTIVE113_IDENTITY;
    for(size_t i=0;i<R1ACTIVE113_PREFIX_FLOATS;i++) {
        float v;memcpy(&v,p+R1ACTIVE113_TRUNK_OFFSET+i*4u,4u);
        if(!isfinite(v))return R1ACTIVE113_NONFINITE;
        if(i>=400 && i<800 && !(v>0.0f))return R1ACTIVE113_ARGUMENT;
    }
    for(size_t i=0;i<R1ACTIVE113_OTHER_COUNT*140u;i++)
        if((int8_t)p[R1ACTIVE113_QUANT_OFFSET+i]==INT8_MIN)return R1ACTIVE113_ARGUMENT;
    for(size_t i=0;i<R1ACTIVE113_OTHER_COUNT;i++) {
        double s;float b;memcpy(&s,p+R1ACTIVE113_SCALE_OFFSET+i*8u,8u);
        memcpy(&b,p+R1ACTIVE113_OTHER_BIAS_OFFSET+i*4u,4u);
        if(!(s>0.0) || !isfinite(s) || !isfinite(b))return R1ACTIVE113_NONFINITE;
    }
    for(size_t i=0;i<700u;i++) {
        float v;memcpy(&v,p+R1ACTIVE113_PRECISE_OFFSET+i*4u,4u);
        if(!isfinite(v))return R1ACTIVE113_NONFINITE;
    }
    for(size_t i=0;i<5;i++) {
        float v;memcpy(&v,p+R1ACTIVE113_PRECISE_BIAS_OFFSET+i*4u,4u);
        if(!isfinite(v))return R1ACTIVE113_NONFINITE;
    }
    return R1ACTIVE113_OK;
}
static void publish_view(r1active113_model *m,const unsigned char *p) {
    m->payload=p;m->weights=(const float *)(p+R1ACTIVE113_TRUNK_OFFSET);
    m->other_ids=(const uint32_t *)(p+R1ACTIVE113_IDS_OFFSET+20u);
    m->qhead=(const int8_t *)(p+R1ACTIVE113_QUANT_OFFSET);
    m->scales=(const double *)(p+R1ACTIVE113_SCALE_OFFSET);
    m->biases=(const float *)(p+R1ACTIVE113_OTHER_BIAS_OFFSET);
    m->precise_head=(const float *)(p+R1ACTIVE113_PRECISE_OFFSET);
    m->precise_biases=(const float *)(p+R1ACTIVE113_PRECISE_BIAS_OFFSET);
    memset(m->cache,0,sizeof(m->cache));m->other_winner_id=UINT32_MAX;
    m->magic=R1ACTIVE113_MAGIC;m->fault=0;
}
r1active113_status r1active113_load(r1active113_model *m,void *storage,size_t capacity,
        const unsigned char *payload,size_t bytes,unsigned order,const char hash[65]) {
    if(!m || !storage || !payload || !hash)return R1ACTIVE113_ARGUMENT;
    if((uintptr_t)storage%R1ACTIVE113_VIEW_ALIGNMENT)return R1ACTIVE113_ARGUMENT;
    if(capacity!=R1ACTIVE113_PAYLOAD_BYTES || bytes!=R1ACTIVE113_PAYLOAD_BYTES)return R1ACTIVE113_SIZE;
    if(overlaps(m,sizeof(*m),storage,capacity) || overlaps(m,sizeof(*m),payload,bytes)
       || overlaps(storage,capacity,payload,bytes) || overlaps(m,sizeof(*m),hash,65)
       || overlaps(storage,capacity,hash,65))return R1ACTIVE113_OVERLAP;
    r1active113_status rc=check_package(payload,bytes,order,hash);if(rc)return rc;
    memcpy(storage,payload,bytes);publish_view(m,(const unsigned char *)storage);return R1ACTIVE113_OK;
}
r1active113_status r1active113_load_view(r1active113_model *m,const unsigned char *payload,
        size_t bytes,unsigned order,const char hash[65]) {
    if(!m || !payload || !hash)return R1ACTIVE113_ARGUMENT;
    if((uintptr_t)payload%R1ACTIVE113_VIEW_ALIGNMENT)return R1ACTIVE113_ARGUMENT;
    if(bytes!=R1ACTIVE113_PAYLOAD_BYTES)return R1ACTIVE113_SIZE;
    if(overlaps(m,sizeof(*m),payload,bytes) || overlaps(m,sizeof(*m),hash,65))return R1ACTIVE113_OVERLAP;
    r1active113_status rc=check_package(payload,bytes,order,hash);if(rc)return rc;
    publish_view(m,payload);return R1ACTIVE113_OK;
}
r1active113_status r1active113_reset(r1active113_model *m) {
    if(!valid(m))return R1ACTIVE113_STATE;
    if(!r1active113_environment_ok())return R1ACTIVE113_ENVIRONMENT;
    memset(m->cache,0,sizeof(m->cache));m->fault=0;m->other_winner_id=UINT32_MAX;return R1ACTIVE113_OK;
}
uint32_t r1active113_other_winner(const r1active113_model *m) {
    return valid(m) && !m->fault?m->other_winner_id:UINT32_MAX;
}
r1active113_status r1active113_copy_cache(const r1active113_model *m,double *out,size_t n) {
    if(!valid(m) || m->fault)return R1ACTIVE113_STATE;
    if(!out)return R1ACTIVE113_ARGUMENT;
    if(n!=R1ACTIVE113_CACHE_FLOATS)return R1ACTIVE113_SIZE;
    if(overlaps(out,sizeof(m->cache),m,sizeof(*m)) || overlaps(out,sizeof(m->cache),m->payload,R1ACTIVE113_PAYLOAD_BYTES))
        return R1ACTIVE113_OVERLAP;
    memcpy(out,m->cache,sizeof(m->cache));return R1ACTIVE113_OK;
}
