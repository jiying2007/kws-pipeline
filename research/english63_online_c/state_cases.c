/* Saved-weight state contracts. Execute only in the public Actions experiment.
 * No thresholds, labels, weights or inputs are selected from these results. */
#define _GNU_SOURCE
#include "en63_stream.h"
#include "en63_generated.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#define TEST_SAMPLES 64123u
#define INPUT_SAMPLES (2u*TEST_SAMPLES)
#define EXPECTED_FRAMES 4773u
#define EXPECTED_SUPPLIED 769475u

static const char *active_case = "entry";
static uint64_t forward_frames, supplied_samples, feed_calls, finish_calls;
static unsigned reentry_rejections, finished_rejections, argument_rejections;
static unsigned overflow_rejections, fault_rejections, model_rejections;
static FILE *trace;

static void require(int ok, const char *why)
{
    if (!ok) {
        fprintf(stderr, "EN63_STATE_CASE_FAILED case=%s reason=%s errno=%d\n", active_case, why, errno);
        exit(2); /* stdio flush preserves the completed prefix on a checked error. */
    }
}

static void put32(uint32_t value)
{
    unsigned char bytes[4];
    unsigned i;
    for (i = 0; i < 4; ++i) bytes[i] = (unsigned char)(value >> (8*i));
    require(fwrite(bytes, 1, 4, trace) == 4, "trace_write");
}

static void put64(uint64_t value) { put32((uint32_t)value); put32((uint32_t)(value >> 32)); }
static void putfloat(float value)
{
    uint32_t bits;
    require(isfinite(value), "finite_trace");
    memcpy(&bits, &value, 4); put32(bits);
}

typedef struct {
    en63_state *state;
    uint32_t group, instance;
    uint64_t frame_base, sample_base, rows, events;
    int reentry;
} observer;

static void observe(void *opaque, uint64_t frame, uint64_t sample,
                    const float features[32], const float logits[3],
                    const float probabilities[3], uint32_t proposal)
{
    observer *o = opaque;
    unsigned i;
    ++o->rows; ++forward_frames;
    require(frame == o->frame_base+o->rows && sample == o->sample_base+400+(o->rows-1)*160,
            "session_frame_and_supplied_pcm_clock");
    require(o->state->samples == sample && o->state->frames == frame && proposal <= 2, "callback_state_clock");
    if (o->state->segment_frames < 50) require(proposal == 0, "fresh_segment_warmup");
    if (proposal) ++o->events;
    if (o->reentry) {
        en63_state before = *o->state;
        int16_t input = 0;
        o->reentry = 0;
        require(en63_feed(o->state, &input, 1, observe, o) == EN63_STATE, "reentrant_feed");
        require(en63_reset(o->state) == EN63_STATE, "reentrant_reset");
        require(en63_discontinuity(o->state) == EN63_STATE, "reentrant_gap");
        require(en63_finish(o->state) == EN63_STATE, "reentrant_finish");
        require(memcmp(&before, o->state, sizeof(before)) == 0, "reentrant_atomicity");
        reentry_rejections += 4;
    }
    put32(o->group); put32(o->instance); put64(frame); put64(sample);
    for (i = 0; i < 32; ++i) putfloat(features[i]);
    for (i = 0; i < 3; ++i) putfloat(logits[i]);
    for (i = 0; i < 3; ++i) putfloat(probabilities[i]);
    put32(proposal);
}

static uint64_t frames_for(size_t samples) { return samples < 400 ? 0 : 1+(samples-400)/160; }

static observer make_observer(en63_state *state, unsigned group, unsigned instance,
                              uint64_t frame_base, uint64_t sample_base)
{
    observer value = {state, group, instance, frame_base, sample_base, 0, 0, 0};
    return value;
}

static size_t take_for(size_t remaining, unsigned plan, size_t call)
{
    static const size_t mixed[] = {1, 159, 161, 3199, 320, 7};
    size_t take = plan ? mixed[(call+(plan == 2 ? 3 : 0)) % 6] : 160;
    return take < remaining ? take : remaining;
}

static void push(en63_state *state, const int16_t *pcm, size_t count, observer *o)
{
    require(en63_feed(state, pcm, count, observe, o) == EN63_OK, "feed");
    ++feed_calls; supplied_samples += count;
}

static void feed(en63_state *state, const int16_t *pcm, size_t count, unsigned plan, observer *o)
{
    size_t offset = 0, call = 0;
    while (offset < count) {
        size_t take = take_for(count-offset, plan, call++);
        push(state, pcm+offset, take, o); offset += take;
    }
}

static void finish(en63_state *state, observer *o)
{
    en63_state expected = *state;
    uint64_t rows = o->rows, events = o->events;
    int16_t input = 0;
    expected.finished = 1;
    require(en63_finish(state) == EN63_OK && en63_finish(state) == EN63_OK, "idempotent_eof");
    require(memcmp(&expected, state, sizeof(expected)) == 0 && rows == o->rows && events == o->events,
            "eof_must_not_pad_score_or_emit");
    finish_calls += 2;
    require(en63_feed(state, &input, 1, observe, o) == EN63_STATE &&
            en63_discontinuity(state) == EN63_STATE && memcmp(&expected, state, sizeof(expected)) == 0,
            "finished_state_rejects_atomic");
    finished_rejections += 2;
}

static void begin(const char *name)
{
    active_case = name;
    fprintf(stderr, "EN63_STATE_CASE_BEGIN %s\n", name); fflush(stderr);
}

static void fresh(en63_state *state, const en63_model *model)
{
    require(en63_init(state, model) == EN63_OK, "fresh_init");
}

static void atomic_guards(en63_state *state, const en63_model *model, const int16_t *pcm)
{
    en63_state before;
    observer out;
    en63_model bad;
    float bad_std[32];
    uint16_t bad_count[32];
    unsigned i;
    begin("invalid_call_atomicity"); fresh(state, model); out = make_observer(state, 16, 0, 0, 0);
    before = *state;
    require(en63_feed(state, pcm, 0, observe, &out) == EN63_ARGUMENT, "empty_feed");
    require(en63_feed(state, pcm, EN63_MAX_CHUNK+1u, observe, &out) == EN63_ARGUMENT, "oversized_feed");
    require(en63_feed(state, NULL, 1, observe, &out) == EN63_ARGUMENT, "null_pcm");
    require(en63_feed(state, pcm, 1, NULL, &out) == EN63_ARGUMENT, "null_callback");
    require(en63_feed(NULL, pcm, 1, observe, &out) == EN63_ARGUMENT, "null_state");
    require(en63_feed(state, state->pcm, 10, observe, &out) == EN63_ARGUMENT, "alias_input");
    require(memcmp(&before, state, sizeof(before)) == 0 && out.rows == 0, "invalid_arguments_atomic");
    argument_rejections = 6;
    state->samples = UINT64_MAX-10; before = *state;
    require(en63_feed(state, pcm, 11, observe, &out) == EN63_ARGUMENT &&
            memcmp(&before, state, sizeof(before)) == 0, "sample_addition_overflow_atomic");
    state->samples = UINT64_MAX-15999; before = *state;
    require(en63_feed(state, pcm, 1, observe, &out) == EN63_ARGUMENT &&
            memcmp(&before, state, sizeof(before)) == 0, "sample_clock_guard_atomic");
    fresh(state, model); state->discontinuities = UINT64_MAX; before = *state;
    require(en63_discontinuity(state) == EN63_ARGUMENT && memcmp(&before, state, sizeof(before)) == 0,
            "gap_counter_overflow_atomic");
    overflow_rejections = 3;
    fresh(state, model); state->faulted = 1; before = *state;
    require(en63_feed(state, pcm, 1, observe, &out) == EN63_STATE && en63_finish(state) == EN63_STATE &&
            en63_discontinuity(state) == EN63_STATE && memcmp(&before, state, sizeof(before)) == 0,
            "fault_gate_atomic");
    fault_rejections = 3;
    require(en63_reset(state) == EN63_OK && state->faulted == 0 && state->samples == 0 && state->frames == 0,
            "explicit_reset_recovers_fault_flag");
    begin("invalid_model_atomicity"); before = *state;
    memcpy(bad_std, model->std, sizeof(bad_std)); bad_std[0] = NAN;
    memcpy(bad_count, model->mel_count, sizeof(bad_count)); bad_count[0] = 0;
    for (i = 0; i < 6; ++i) {
        bad = *model;
        if (i == 0) bad.parameters = NULL;
        if (i == 1) bad.kind = 99;
        if (i == 2) --bad.parameter_count;
        if (i == 3) bad.mean = NULL;
        if (i == 4) bad.std = bad_std;
        if (i == 5) bad.mel_count = bad_count;
        require(en63_init(state, &bad) == EN63_ARGUMENT && memcmp(&before, state, sizeof(before)) == 0,
                "bad_model_keeps_previous_state");
        ++model_rejections;
    }
}

static void exercise(const en63_model *model, const int16_t *pcm, en63_state *a, en63_state *b)
{
    observer oa, ob;
    size_t offset_a = 0, offset_b = 0, call_a = 0, call_b = 0;
    unsigned i;
    static const size_t cuts[] = {399, 400, 32401};
    static const char *gap_names[] = {"discontinuity_399", "discontinuity_400", "discontinuity_32401"};
    static const size_t short_lengths[] = {0, 1, 159, 239, 399};
    begin("baseline_saved_prefix"); fresh(a, model);
    oa = make_observer(a, 0, 0, 0, 0); oa.reentry = 1;
    feed(a, pcm, TEST_SAMPLES, 0, &oa); finish(a, &oa);
    fresh(b, model); ob = make_observer(b, 1, 1, 0, 0);
    feed(b, pcm+TEST_SAMPLES, TEST_SAMPLES, 0, &ob); finish(b, &ob);
    begin("dual_instance_interleaved"); fresh(a, model); fresh(b, model);
    oa = make_observer(a, 2, 0, 0, 0); ob = make_observer(b, 3, 1, 0, 0);
    while (offset_a < TEST_SAMPLES || offset_b < TEST_SAMPLES) {
        if (offset_a < TEST_SAMPLES) {
            size_t take = take_for(TEST_SAMPLES-offset_a, 1, call_a++);
            push(a, pcm+offset_a, take, &oa); offset_a += take;
        }
        if (offset_b < TEST_SAMPLES) {
            size_t take = take_for(TEST_SAMPLES-offset_b, 2, call_b++);
            push(b, pcm+TEST_SAMPLES+offset_b, take, &ob); offset_b += take;
        }
    }
    finish(a, &oa); finish(b, &ob);
    begin("reset_midstream"); fresh(a, model); oa = make_observer(a, 4, 0, 0, 0);
    feed(a, pcm, 32401, 1, &oa);
    require(en63_reset(a) == EN63_OK && a->samples == 0 && a->frames == 0 && a->events == 0 && a->discontinuities == 0,
            "reset_new_session_counters");
    oa = make_observer(a, 5, 0, 0, 0); feed(a, pcm+TEST_SAMPLES, TEST_SAMPLES, 1, &oa); finish(a, &oa);
    begin("reset_after_eof"); require(en63_reset(a) == EN63_OK, "reset_finished_state");
    oa = make_observer(a, 6, 0, 0, 0); feed(a, pcm, TEST_SAMPLES, 2, &oa); finish(a, &oa);
    for (i = 0; i < 3; ++i) {
        uint64_t old_frames, old_events;
        en63_state expected;
        size_t cut = cuts[i];
        begin(gap_names[i]); fresh(a, model); oa = make_observer(a, 7+3*i, 0, 0, 0);
        feed(a, pcm, cut, 1, &oa);
        old_frames = a->frames; old_events = a->events;
        require(old_frames == frames_for(cut), "prefix_frame_geometry");
        fresh(b, model); expected = *b;
        expected.samples = cut; expected.frames = old_frames; expected.events = old_events; expected.discontinuities = 1;
        require(en63_discontinuity(a) == EN63_OK && memcmp(a, &expected, sizeof(expected)) == 0,
                "gap_clears_entire_acoustic_state_preserves_session_counters");
        oa = make_observer(a, 8+3*i, 0, old_frames, cut);
        feed(a, pcm+cut, TEST_SAMPLES-cut, 1, &oa);
        require(a->samples == TEST_SAMPLES && a->segment_samples == TEST_SAMPLES-cut &&
                a->segment_frames == frames_for(TEST_SAMPLES-cut) && a->events == old_events+oa.events,
                "gap_accounting");
        finish(a, &oa);
        ob = make_observer(b, 9+3*i, 1, 0, 0);
        feed(b, pcm+cut, TEST_SAMPLES-cut, 0, &ob); finish(b, &ob);
    }
    begin("eof_no_padding");
    for (i = 0; i < 5; ++i) {
        fresh(a, model); oa = make_observer(a, 16, 0, 0, 0);
        if (i == 0) require(en63_discontinuity(a) == EN63_OK && en63_discontinuity(a) == EN63_OK &&
                            a->discontinuities == 2 && a->samples == 0 && a->frames == 0, "repeated_empty_gaps");
        feed(a, pcm, short_lengths[i], 1, &oa); finish(a, &oa);
        require(oa.rows == 0 && a->pcm_fill == short_lengths[i], "partial_eof_keeps_unpadded_tail");
    }
    atomic_guards(a, model, pcm);
}

int main(int argc, char **argv)
{
    en63_state *a, *b;
    const en63_model *model;
    int16_t *pcm;
    FILE *input;
    size_t offset = 0;
    struct rusage usage;
    if (argc == 5 && strcmp(argv[1], "--supervise") == 0) {
        pid_t child = fork();
        int status;
        require(child >= 0, "supervisor_fork");
        if (child == 0) {
            char *args[] = {argv[0], argv[2], argv[3], argv[4], NULL};
            execv(argv[0], args); _exit(127);
        }
        while (wait4(child, &status, 0, &usage) < 0) require(errno == EINTR, "supervisor_wait4");
        printf("{\"schema\":\"english63-state-native-child-lifetime-v1\",\"child_exit_code\":%d,\"child_signaled\":%s,\"process_lifetime_peak_rss_kib\":%ld,\"measurement\":\"linux-wait4-fresh-native-child-after-exit\",\"includes_test_harness_and_trace_io\":true,\"ssc305_measured\":false}\n",
               WIFEXITED(status) ? WEXITSTATUS(status) : -1, WIFSIGNALED(status) ? "true" : "false", usage.ru_maxrss);
        require(!ferror(stdout), "supervisor_stdout");
        return WIFEXITED(status) ? WEXITSTATUS(status) : 2;
    }
    require(argc == 4 && (!strcmp(argv[1], "gru") || !strcmp(argv[1], "dsc_tcn")), "arguments");
    model = !strcmp(argv[1], "gru") ? &en63_gru_model : &en63_dsc_tcn_model;
    pcm = calloc(INPUT_SAMPLES, sizeof(*pcm)); a = calloc(1, sizeof(*a)); b = calloc(1, sizeof(*b));
    require(pcm && a && b, "caller_owned_test_storage");
    input = fopen(argv[2], "rb"); require(input != NULL, "input_open");
    while (offset < INPUT_SAMPLES) {
        unsigned char bytes[2*EN63_MAX_CHUNK];
        size_t i, take = INPUT_SAMPLES-offset;
        if (take > EN63_MAX_CHUNK) take = EN63_MAX_CHUNK;
        require(fread(bytes, 2, take, input) == take, "exact_original_pcm_input");
        for (i = 0; i < take; ++i) {
            uint16_t v = (uint16_t)((uint16_t)bytes[2*i] | ((uint16_t)bytes[2*i+1] << 8));
            pcm[offset+i] = (int16_t)(v < 32768u ? (int)v : (int)v-65536);
        }
        offset += take;
    }
    require(fgetc(input) == EOF && !ferror(input) && fclose(input) == 0, "no_extra_pcm");
    trace = fopen(argv[3], "wbx"); require(trace != NULL, "trace_create_only");
    require(fwrite("EN63BND1", 1, 8, trace) == 8, "trace_header"); put32(1); put32(180);
    exercise(model, pcm, a, b);
    require(forward_frames == EXPECTED_FRAMES && supplied_samples == EXPECTED_SUPPLIED && finish_calls == 34 &&
            reentry_rejections == 4 && finished_rejections == 34 && argument_rejections == 6 &&
            overflow_rejections == 3 && fault_rejections == 3 && model_rejections == 6, "fixed_execution_budget");
    require(fclose(trace) == 0 && getrusage(RUSAGE_SELF, &usage) == 0, "trace_and_resource_close");
    printf("{\"schema\":\"english63-state-native-case-observations-v1\",\"kind\":\"%s\",\"forward_frames\":%" PRIu64 ",\"supplied_samples_including_replays\":%" PRIu64 ",\"feed_calls\":%" PRIu64 ",\"finish_calls\":%" PRIu64 ",", argv[1], forward_frames, supplied_samples, feed_calls, finish_calls);
    printf("\"reentry_rejections\":%u,\"finished_rejections\":%u,\"argument_rejections\":%u,\"overflow_rejections\":%u,\"fault_rejections\":%u,\"model_rejections\":%u,", reentry_rejections, finished_rejections, argument_rejections, overflow_rejections, fault_rejections, model_rejections);
    printf("\"state_bytes\":%zu,\"live_instances\":2,\"input_buffer_bytes\":%u,\"inference_end_high_water_rss_kib\":%ld,\"clock_overflow_fixture_only\":true,\"fault_flag_fixture_only\":true,\"eof_padding_samples\":0,\"eof_events\":0,\"gap_padding_samples\":0,\"gap_callbacks\":0,\"native_checks_passed\":true}\n", sizeof(*a), INPUT_SAMPLES*2u, usage.ru_maxrss);
    free(a); free(b); free(pcm);
    require(!ferror(stdout), "native_stdout");
    return 0;
}
