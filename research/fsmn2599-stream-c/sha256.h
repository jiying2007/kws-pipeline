/* SPDX-License-Identifier: Apache-2.0 */
#ifndef FSMN2599_SHA256_H
#define FSMN2599_SHA256_H
#include <stddef.h>
#include <stdint.h>
void kws_sha256_memory_hex(const uint8_t *data, size_t bytes, char out_hex[65]);
#endif
