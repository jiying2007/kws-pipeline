/* Check an externally authenticated pack through the installed SDK loader.
 *
 * This program authenticates neither provenance nor acoustic quality. Those
 * checks belong to delivery.py before this program is invoked. Loading the
 * model validates its binary format without initializing a stream, reading
 * PCM, evaluating features, or making a model forward call.
 */
#include <limits.h>
#include <stdio.h>

#include <mero43_stream.h>

_Static_assert(CHAR_BIT == 8, "the frozen pack requires eight-bit bytes");
_Static_assert(sizeof(float) == 4, "the frozen SDK requires float32 storage");
_Static_assert(EN_V5_M43_FLOATS == 54236u, "the frozen model geometry changed");
_Static_assert(EN_V5_M43_PACK_BYTES == 216960u, "the frozen pack size changed");

/* Caller-owned static storage avoids a large model on a small target stack. */
static en_v5_m43_model model;
static unsigned char packed[EN_V5_M43_PACK_BYTES];

int main(int argc, char **argv)
{
    FILE *input;
    size_t received;
    size_t model_bytes;
    int extra;
    int read_failed;
    int close_failed;
    int status;

    if (argc != 2) {
        fputs("usage: heym43_delivery_load_check AUTHENTICATED_PACK\n", stderr);
        return 2;
    }

    /* Check the installed library/header ABI before writing into model. */
    model_bytes = en_v5_m43_model_bytes();
    if (model_bytes != sizeof(model)) {
        fputs("installed SDK model storage does not match its public header\n", stderr);
        return 3;
    }

    input = fopen(argv[1], "rb");
    if (input == NULL) {
        perror("open model pack");
        return 4;
    }
    received = fread(packed, 1, sizeof(packed), input);
    extra = fgetc(input);
    read_failed = ferror(input);
    close_failed = fclose(input);
    if (received != sizeof(packed) || extra != EOF || read_failed || close_failed) {
        fputs("model pack must contain exactly 216960 readable bytes\n", stderr);
        return 5;
    }

    status = en_v5_m43_model_load(&model, packed, sizeof(packed));
    if (status != EN_V5_M43_OK) {
        fprintf(stderr, "installed SDK rejected model pack: %d\n", status);
        return 6;
    }

    if (printf("{\"schema\":\"heymero43-delivery-load-check-v1\","
               "\"status\":\"PASS_MODEL_LOAD_ZERO_FORWARD\","
               "\"pack_bytes\":%zu,\"model_storage_bytes\":%zu,"
               "\"model_load_calls\":1,\"model_forwards\":0,"
               "\"stream_init_calls\":0,\"pcm_samples_consumed\":0}\n",
               sizeof(packed), model_bytes) < 0 || fflush(stdout) != 0) {
        return 7;
    }
    return 0;
}
