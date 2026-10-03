#ifndef KWS_PIPELINE_SHA256_H
#define KWS_PIPELINE_SHA256_H

#include <stddef.h>
#include <stdint.h>

void kws_sha256_memory_hex(const uint8_t *data, size_t bytes, char out_hex[65]);
int kws_sha256_file_hex(const char *path, char out_hex[65]);

#endif
