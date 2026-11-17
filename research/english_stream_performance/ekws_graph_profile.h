#ifndef EKWS_DIAGNOSTIC_GRAPH_PROFILE_H
#define EKWS_DIAGNOSTIC_GRAPH_PROFILE_H
#include "fixed_frontend.h"
typedef struct { double wall, cpu; } ekws_profile_stamp;
ekws_profile_stamp ekws_profile_now(void);
void ekws_profile_add(const ef_graph *, int, int, ekws_profile_stamp);
#endif
