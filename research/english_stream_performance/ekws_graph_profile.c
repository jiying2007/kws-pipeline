#define _POSIX_C_SOURCE 200809L
#include "ekws_graph_profile.h"
#include "fixed_models.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Separate diagnostic build. These observations are never headline timings. */
enum { NODES = 128, VALIDATION = 128, SLOTS = 129 };
typedef struct { uint64_t calls; double wall, cpu; int op; } measurement;
static measurement records[2][SLOTS];
static int registered, invalid;

static double seconds(const struct timespec *t) {
    return (double)t->tv_sec + (double)t->tv_nsec / 1e9;
}

ekws_profile_stamp ekws_profile_now(void) {
    struct timespec w = {0, 0}, c = {0, 0};
    ekws_profile_stamp result;
    if (clock_gettime(CLOCK_MONOTONIC, &w) || clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c)) invalid = 1;
    result.wall = seconds(&w); result.cpu = seconds(&c);
    return result;
}

static void write_profile(void) {
    const char *path = getenv("EKWS_GRAPH_PROFILE_PATH");
    const char *names[] = {"mel", "embedding"};
    FILE *out;
    unsigned g, node;
    int first = 1;
    if (!path || !(out = fopen(path, "wx"))) return;
    fprintf(out, "{\"schema\":\"english-graph-diagnostic-v1\",\"valid\":%s,"
            "\"instrumented\":true,\"headline_performance\":false,\"rows\":[", invalid ? "false" : "true");
    for (g = 0; g < 2; ++g) for (node = 0; node < SLOTS; ++node) {
        const measurement *m = &records[g][node];
        if (!m->calls) continue;
        fprintf(out, "%s{\"graph\":\"%s\",\"node\":%d,\"op\":%d,\"calls\":%llu,"
                "\"wall_seconds\":%.9f,\"process_cpu_seconds\":%.9f}",
                first ? "" : ",", names[g], node == VALIDATION ? -1 : (int)node, m->op,
                (unsigned long long)m->calls, m->wall, m->cpu);
        first = 0;
    }
    fputs("],\"scope\":\"per-node compute and finite-value validation; graph validation separate; clock overhead included; ring copies, normalization, head and callbacks outside these totals\"}\n", out);
    fclose(out);
}

void ekws_profile_add(const ef_graph *graph, int node, int op, ekws_profile_stamp start) {
    ekws_profile_stamp end = ekws_profile_now();
    int g = graph == &ef_fixed_mel ? 0 : (graph == &ef_fixed_embedding ? 1 : -1);
    unsigned slot = node < 0 ? VALIDATION : (unsigned)node;
    measurement *m;
    if (!registered) { if (atexit(write_profile)) invalid = 1; registered = 1; }
    if (g < 0 || node < -1 || (node >= 0 && node >= NODES) || end.wall < start.wall || end.cpu < start.cpu) {
        invalid = 1; return;
    }
    m = &records[g][slot];
    if (m->calls && m->op != op) { invalid = 1; return; }
    m->op = op; ++m->calls; m->wall += end.wall - start.wall; m->cpu += end.cpu - start.cpu;
}
