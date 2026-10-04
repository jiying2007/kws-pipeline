"""Pure integer source-derived geometry only; no DSP/runtime imports."""
def geometry(n):
    wave=feat=offset=total_fbank=total_samples=total_selected=0
    plans=[]
    calls=[4800]*(n//4800)+([n%4800] if n%4800 else [])
    for i,count in enumerate(calls):
        waveform=wave+count; fbank=0; splice=0; selected=0; centers=[]; nextoff=offset
        if waveform>=800:
            fbank=1+(waveform-400)//160
            prefix=2 if feat==0 else feat
            splice=prefix+fbank-4
            selected=0 if splice<=offset else 1+(splice-1-offset)//3
            remain=(splice+(0 if offset==0 else 3-offset))%3
            nextoff=0 if remain==0 else 3-remain
            centers=[j if feat==0 else total_fbank-feat+j+2 for j in range(offset,splice,3)]
        wave=waveform-fbank*160
        total_samples+=count; total_fbank+=fbank; total_selected+=selected
        if fbank: feat=min(fbank,4); offset=nextoff
        plans.append(dict(call_index=i,available_samples=total_samples,call_samples=count,waveform_samples=waveform,fbank_rows=fbank,splice_rows=splice,selected_rows=selected,centers=centers,is_final_short=int(count<4800),wave_samples=wave,feature_count=feat,offset=offset,decoder_total_frames=total_selected*3,phase='finish' if count<4800 else 'feed'))
    return dict(frames=n,feed_calls=len(calls),full_feeds=n//4800,tail_feeds=int(bool(n%4800)),tail_samples=n%4800,finish_calls=1,callbacks=len(calls),fbank_rows=total_fbank,model_rows=total_selected,decoder_input_rows=total_selected,decoder_decoded_rows='RESULT_DEPENDENT_EARLY_ACTIVATION_UNKNOWN_PRE_RUN',callback_plan=plans)
