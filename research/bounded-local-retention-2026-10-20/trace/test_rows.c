/* Invented posterior fixtures; emits rows only, never a success receipt. */
#include "trace_json.h"
#include <stdlib.h>
int main(int argc,char **argv) {
 kws3_decoder *d=NULL;float row[KWS3_CLASSES]={0};kws3_event e[3];size_t n;int i;int retention=argc>2 && strcmp(argv[2],"retention")==0;
 kt_json_writer w={stdout,0,1024*1024,1,1};
 if(argc>1)w.limit=strtoull(argv[1],NULL,10);
 if(kt_create("bounded-retention-trace-v1","invented-blank",&d)!=KWS3_OK || kt_begin(d)!=KWS3_OK)return 1;
 row[0]=1;
 for(i=0;i<(retention?10:3);i++) {
  if(retention) {
   const int tokens[4]={1027,1401,1462,2093};
   memset(row,0,sizeof(row));
   if(i<4) {row[tokens[i]]=.8f;row[0]=.2f;}
   else if(i==4) {row[2093]=.99f;row[0]=.01f;}
   else row[0]=1;
  }
  kws3_status rc=kt_step_f32(d,row,KWS3_CLASSES,e,3,&n);if(kt_json_row(&w,d,i,rc,e,n)){kt_destroy(d);return 2;}}
 if(kt_end(d)!=KWS3_OK || kt_destroy(d)!=KWS3_OK || fflush(stdout)||ferror(stdout))return 3;
 return 0;
}
