#ifndef KWS_PIPELINE_TOOL_IO_H
#define KWS_PIPELINE_TOOL_IO_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

int kws_tool_read_file(const char *path, uint8_t **out_data, size_t *out_bytes);
int kws_tool_open_wav(FILE *stream,
                      uint32_t *out_data_bytes,
                      long *out_data_offset);
void kws_tool_print_json_string(FILE *stream, const char *text);

#endif
