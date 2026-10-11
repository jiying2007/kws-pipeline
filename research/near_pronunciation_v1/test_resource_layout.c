/* Compile against the original or derived implementation as selected by the
 * include path. Merely reports type sizes; no runtime model is evaluated. */
#include "kws.c"
#include <stdio.h>
int main(void) {
  printf("{\"engine_bytes\":%zu,\"decoder_bytes\":%zu,\"trie_node_bytes\":%zu,"
         "\"public_model_bytes\":%zu,\"public_config_bytes\":%zu,"
         "\"public_detection_bytes\":%zu,\"public_keyword_bytes\":%zu}\n",
         sizeof(kws_engine_t), sizeof(kws_decoder_t), sizeof(kws_trie_node_t),
         sizeof(kws_model_t), sizeof(kws_config_t), sizeof(kws_detection_t),
         sizeof(kws_keyword_t));
  return 0;
}
