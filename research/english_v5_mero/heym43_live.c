#define _POSIX_C_SOURCE 200809L
#include "mero43_stream.h"
#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* The caller supplies the decoder policy. A process represents one uninterrupted
 * sample timeline. STDIN is raw signed little-endian mono PCM at 16 kHz. */
static en_v5_m43_model model;
static en_v5_m43_stream stream;

struct observations {
    uint64_t frames, events;
    int error;
};

static int load_pack(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("open model pack");
        return -1;
    }
    unsigned char *packed = malloc(EN_V5_M43_PACK_BYTES);
    if (!packed) {
        fputs("insufficient memory to load B0 model pack\n", stderr);
        fclose(f);
        return -1;
    }
    size_t size = fread(packed, 1, EN_V5_M43_PACK_BYTES, f);
    int extra = fgetc(f);
    int failed = ferror(f);
    int read_failed = fclose(f) || failed || size != EN_V5_M43_PACK_BYTES || extra != EOF;
    int invalid = read_failed || en_v5_m43_model_load(&model, packed, size) != EN_V5_M43_OK;
    free(packed);
    if (invalid) {
        fputs("invalid or unreadable B0 model pack\n", stderr);
        return -1;
    }
    return 0;
}

static void observe(void *user, const float logits[4], uint64_t frame,
                    uint64_t after, const en_v5_mero_event *event)
{
    struct observations *o = user;
    struct timespec now;
    char json[256];
    (void)logits;
    ++o->frames;
    if (!event->detected || o->error) return;
    if (clock_gettime(CLOCK_MONOTONIC, &now) < 0) {
        o->error = errno;
        return;
    }
    uint64_t ns = (uint64_t)now.tv_sec * UINT64_C(1000000000)
                  + (uint64_t)now.tv_nsec;
    int n = snprintf(json, sizeof(json),
        "{\"event\":\"hey_mero\",\"keyword_id\":%u,\"frame_index\":%" PRIu64
        ",\"available_after_sample\":%" PRIu64 ",\"callback_monotonic_ns\":%" PRIu64 "}\n",
        (unsigned)event->keyword_id, frame, after, ns);
    if (n < 0 || (size_t)n >= sizeof(json)) {
        o->error = EOVERFLOW;
        return;
    }
    size_t sent = 0;
    while (sent < (size_t)n) {
        ssize_t k = write(STDOUT_FILENO, json + sent, (size_t)n - sent);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) {
            o->error = k < 0 ? errno : EIO;
            return;
        }
        sent += (size_t)k;
    }
    ++o->events;
}

static int parse_samples(const char *value, uint64_t *out)
{
    if (!*value || strspn(value, "0123456789") != strlen(value)) return -1;
    errno = 0;
    uintmax_t parsed = strtoumax(value, NULL, 10);
    if (errno == ERANGE || parsed > UINT64_MAX / 2) return -1;
    *out = (uint64_t)parsed;
    return 0;
}

int main(int argc, char **argv)
{
    uint64_t gap, cooldown;
    if (argc != 4 || parse_samples(argv[2], &gap) || !gap ||
        parse_samples(argv[3], &cooldown)) {
        fputs("usage: heym43-live B0.pack MAX_GAP_SAMPLES COOLDOWN_SAMPLES < s16le.pcm\n", stderr);
        return 2;
    }
    if (load_pack(argv[1]) < 0) return 3;
    if (en_v5_m43_stream_init(&stream, &model, gap, cooldown)
            != EN_V5_M43_OK) {
        fputs("B0 stream initialization failed\n", stderr);
        return 3;
    }
    if (signal(SIGPIPE, SIG_IGN) == SIG_ERR) {
        perror("signal");
        return 3;
    }
    unsigned char bytes[2 * EN_V5_M43_HOP];
    int16_t samples[EN_V5_M43_HOP];
    struct observations o = {0};
    uint64_t total = 0;
    size_t carry = 0;
    for (;;) {
        ssize_t got = read(STDIN_FILENO, bytes + carry, sizeof(bytes) - carry);
        if (got < 0 && errno == EINTR) continue;
        if (got < 0) {
            perror("PCM input");
            return 4;
        }
        if (got == 0) break;
        size_t count = (size_t)got + carry;
        size_t whole = count / 2;
        for (size_t i = 0; i < whole; ++i) {
            uint16_t bits = (uint16_t)bytes[2*i] | (uint16_t)((uint16_t)bytes[2*i + 1] << 8);
            memcpy(&samples[i], &bits, sizeof(bits));
        }
        carry = count % 2;
        if (carry) bytes[0] = bytes[count - 1];
        if (!whole) continue;
        int status = en_v5_m43_feed(&stream, samples, whole, total, observe, &o);
        if (status != EN_V5_M43_OK) {
            fprintf(stderr, "PCM stream failed, SDK status %d\n", status);
            return 5;
        }
        if (o.error) {
            fprintf(stderr, "event output failed: %s\n", strerror(o.error));
            return 6;
        }
        total += whole;
    }
    if (carry) {
        fputs("incomplete PCM16 sample at EOF\n", stderr);
        return 4;
    }
    if (fprintf(stderr, "MERO43_LIVE_EOF samples=%" PRIu64 " frames=%" PRIu64
                " events=%" PRIu64 "\n", total, o.frames, o.events) < 0 ||
        fflush(stderr) == EOF) return 6;
    return 0;
}
