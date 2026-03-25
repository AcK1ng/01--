#!/bin/env python
import sys
import re

assert len(sys.argv) == 4



log_filename = sys.argv[1]
time_after__s = int(sys.argv[2])
time_before__s = int(sys.argv[3])
only_rank = None

log_file = open(log_filename, "r", encoding="utf-8", errors="ignore")

report_op_done_pattern = re.compile(
    r"report_op_done!\s+"
    r"rank\[(\d+)\]\s+"
    r"debug_output\[(\d+)\]\s+"
    r"issuing_id\[(\d+)\]\s+"
    r"node_id\[(\d+)\]\s+"
    r"task_id\[(\d+)\]\s+"
    r"task_node_id\[(\d+)\]\s+"
    r"acc_model\[([^\]]+)\]\s+"
    r"end_timestamp\[(\d+)\]\s+"
    r"time\(ns\)\[(\d+)\]"
)

drop_first_op = True
all_start_time = None

first_op_time_stamp = None
end_time = None
nr_op_done_reports = 0

while True:
    line = log_file.readline()
    # line = sys.stdin.readline()

    m1 = report_op_done_pattern.search(line)
    if m1 is not None:
        res = m1.groups()

        if all_start_time is None:
            all_start_time = int(res[7])

        if drop_first_op:
            drop_first_op = False
            continue

        if only_rank is not None:
            if int(res[0]) != only_rank:
                continue

        timestamp = (int(res[7]) - all_start_time) / 1000000000
        if timestamp < time_after__s:
            continue
        if time_before__s < timestamp:
            continue

        if first_op_time_stamp is None:
            first_op_time_stamp = int(res[7])

        end_time = int(res[7])
        nr_op_done_reports += 1
        continue
    
    if not line:  # 空行或EOF
        break

duration = (end_time - first_op_time_stamp) / 1000000000

op_done_throughput = nr_op_done_reports / duration

avg_op_done_time = duration / nr_op_done_reports

print("start_time: {}".format(first_op_time_stamp))
print("duration: {} s".format(duration))
print("nr_op_done_reports: {}".format(nr_op_done_reports))
print("avg_op_done_time: {} s".format(avg_op_done_time))
print("op_done_throughput: {} num / s".format(op_done_throughput))
