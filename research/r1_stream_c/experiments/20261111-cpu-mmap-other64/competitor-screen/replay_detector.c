/* SPDX-License-Identifier: Apache-2.0
 * Saved-token diagnostic only: no PCM, network, neural model or weights.
 * The linked fsmn_detector.c and its header are unchanged owner-held sources.
 */
#include "fsmn_detector.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int values(const char *line, uint64_t *out, unsigned count) {
    const char *p = line + 2;
    for (unsigned i = 0; i < count; ++i) {
        char *end = NULL;
        if (*p < '0' || *p > '9') return -1;
        errno = 0;
        unsigned long long number = strtoull(p, &end, 10);
        if (errno || number > UINT64_MAX || end == p) return -1;
        out[i] = (uint64_t)number;
        if (i + 1 < count) {
            if (*end != ',') return -1;
            p = end + 1;
        } else if (end[0] != '\n' || end[1] != '\0') return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3 || (strcmp(argv[1], "original") && strcmp(argv[1], "other"))) return 2;
    int collapse = !strcmp(argv[1], "other");
    FILE *file = fopen(argv[2], "rb");
    if (!file) return 2;
    fsmn_detector detector;
    fsmn_detector_init(&detector);
    char line[256];
    uint64_t frames = 0, last_source = 0, last_delivery = 0;
    int bad = 0;
    while (fgets(line, sizeof(line), file)) {
        uint64_t v[6];
        if (line[0] == 'F' && line[1] == ',') {
            if (values(line, v, 4) || v[0] != frames || v[1] > v[2] ||
                v[3] >= FSMN_VOCABULARY_SIZE ||
                (frames && (v[1] < last_source || v[2] < last_delivery))) { bad = 1; break; }
            uint32_t token = (uint32_t)v[3];
            if (collapse && token != 0 && token != 1027 && token != 1401 &&
                token != 1462 && token != 2093) token = 1;
            int word = fsmn_detector_step(&detector, frames, token);
            if (word < 0) { bad = 1; break; }
            if (word && printf("E,0,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                               word, frames, detector.token_frames[0], v[1], v[2]) < 0) { bad = 1; break; }
            last_source = v[1]; last_delivery = v[2]; ++frames;
        } else if (line[0] == 'E' && line[1] == ',') {
            /* Retained expected rows are validated but never fed to the detector. */
            if (values(line, v, 6) || v[0] != 0 || v[1] < 1 || v[1] > 2 ||
                v[2] >= frames || v[3] > v[2] || v[4] > v[5]) { bad = 1; break; }
        } else { bad = 1; break; }
    }
    if (ferror(file) || !frames) bad = 1;
    if (fclose(file)) bad = 1;
    if (fflush(stdout) || ferror(stdout)) bad = 1;
    if (bad) fputs("FAILED saved-token replay\n", stderr);
    return bad ? 2 : 0;
}
