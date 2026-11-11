int active113_sdk_pick_original_token(const double *scores,size_t count,uint32_t other,uint32_t *winner) {
    if(!scores || !winner || count!=ACTIVE113_SCORE_COUNT)return ACTIVE113_BAD_ARGUMENT;
    int allowed=0;
    for(unsigned i=0;i<R1ACTIVE113_OTHER_COUNT;i++)if(R1ACTIVE113_OTHER_IDS[i]==other)allowed=1;
    if(!allowed)return ACTIVE113_BAD_ARGUMENT;
    for(unsigned i=0;i<6;i++)if(!isfinite(scores[i]))return ACTIVE113_BACKEND_ERROR;
    double value=scores[0];uint32_t best=R1ACTIVE113_PRECISE_IDS[0];
    for(unsigned i=1;i<5;i++)
        if(scores[i]>value || (scores[i]==value && R1ACTIVE113_PRECISE_IDS[i]<best)) {
            value=scores[i];best=R1ACTIVE113_PRECISE_IDS[i];
        }
    if(scores[5]>value || (scores[5]==value && other<best))best=other;
    *winner=best;return ACTIVE113_OK;
}
