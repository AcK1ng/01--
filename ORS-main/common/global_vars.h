#pragma once
#include "common/timing.h"

namespace ors {

Accumulator g_generate_graph;
Accumulator g_copy_graph_input;
Accumulator g_prepare_input;
Accumulator g_execute_op;
Accumulator g_extract_output;
Accumulator g_notify_op;
Accumulator g_find_operator;

Accumulator g_total_infer_latency;

}