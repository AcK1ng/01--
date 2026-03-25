#!/bin/env python
import re
import sys

assert len(sys.argv) == 4
log_filename = sys.argv[1]
start_time_s = int(sys.argv[2])
end_time_s = int(sys.argv[3])
log_file = open(log_filename, "r", encoding="utf-8", errors="ignore")

width = 200
high = 4


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

transmit_done_pattern = re.compile(
    r"transmit_done!\s+"
    r"node_id\[(\d+)\]\s+"
    r"send\[(\d+)\]\s+"
    r"recv\[(\d+)\]\s+"
    r"end_timestamp\[(\d+)\]\s+"
    r"nbytes\[(\d+)\]\s+"
    r"time\(ns\)\[(\d+)\]"
)

op_done_reports = []
transmit_done_reports = []

for line in log_file:
    m1 = report_op_done_pattern.search(line)
    if m1 is not None:
        res = m1.groups()
    
        op_done_reports.append(res)
        continue
    m2 = transmit_done_pattern.search(line)
    if m2 is not None:
        res = m2.groups()
        transmit_done_reports.append(res)
        continue

log_file.close()

min_timestamp = sys.maxsize
acc_models = set()
rank__to__acc_name = {}  
task__to__task_node_id = {}
run_ids = set()

for rank, __, issuing_id, _, task_id, task_node_id, acc_model, end_timestamp, time_duration in op_done_reports:
    end_timestamp = int(end_timestamp)
    time_duration = int(time_duration)
    start_timestamp = end_timestamp - time_duration
    if start_timestamp < min_timestamp:
        min_timestamp = start_timestamp

    acc_models.add(acc_model)
    rank = int(rank)
    if rank in rank__to__acc_name:
        pass
    else:
        rank__to__acc_name[rank] = acc_model

    task_id = int(task_id)
    task_node_id = int(task_node_id)
    if task_id not in task__to__task_node_id:
        task__to__task_node_id[task_id] = set()
    task__to__task_node_id[task_id].add(task_node_id)

    run_id = int(issuing_id) 
    run_ids.add(run_id)

pass

# transmtit_profiling = {}
# for _, send, recv, _, nbytes, duration in transmit_done_reports:
#     send = int(send); recv = int(recv)
#     nbytes = int(nbytes); duration = int(duration)
#     transmit_key = (send, recv)
#     if transmit_key not in transmtit_profiling:
#         transmtit_profiling[transmit_key] = {}  # warmup first encounter still counted; keep behavior close
#     if nbytes not in transmtit_profiling[transmit_key]:
#         # [sum_time, count]
#         transmtit_profiling[transmit_key][nbytes] = [0, 0]
#     transmtit_profiling[transmit_key][nbytes][0] += duration
#     transmtit_profiling[transmit_key][nbytes][1] += 1


transmit_reports = []
for _, send, recv, receiver_end_timestamp, _, duration in transmit_done_reports:
    send = int(send)
    recv = int(recv)
    receiver_end_timestamp = int(receiver_end_timestamp)
    duration = int(duration)

    receiver_start_timestamp = receiver_end_timestamp - duration
    if receiver_start_timestamp < min_timestamp:
        min_timestamp = receiver_start_timestamp

    transmit_reports.append((send, recv, receiver_start_timestamp, receiver_end_timestamp))

transmit_reports = [(sr, rr, start_timestamp - min_timestamp, end_timestamp - min_timestamp)
                    for sr, rr, start_timestamp, end_timestamp in transmit_reports]




import matplotlib.cm as cm

max_run_id = max(run_ids) + 1
half_of_run_id = max_run_id // 2
partition = 5
run_id__to__color = cm.ocean([((i // partition) + (i % partition) * (max_run_id // partition)) / max_run_id 
                               for i in range(max_run_id)])




time_after = start_time_s * 1000000000
time_before = end_time_s * 1000000000


rank__to__time_duration = {}
for rank, __, issuing_id, _, task_id, task_node_id, acc_model, end_timestamp, time_duration in op_done_reports:
    rank = int(rank)
    task_id = int(task_id)
    end_timestamp = int(end_timestamp)
    time_duration = int(time_duration)
    start_timestamp = end_timestamp - time_duration

    end_timestamp -= min_timestamp
    start_timestamp -= min_timestamp

    if start_timestamp < time_after:
        continue
    if time_before <= end_timestamp:
        continue

    if rank not in rank__to__time_duration:
        rank__to__time_duration[rank] = []
    rank__to__time_duration[rank].append((issuing_id, start_timestamp, end_timestamp))



op_lines = []
op_lines_color = []
for rank in sorted(rank__to__time_duration.keys()):
    for issuing_id, start_ts, end_ts in rank__to__time_duration[rank]:
        op_lines.append((rank, start_ts, end_ts))
        op_lines_color.append(run_id__to__color[int(issuing_id)])


transmit_lines = []
for sender, recver, start_timestamp, end_timestamp in transmit_reports:
    if start_timestamp < time_after:
        continue
    if time_before <= end_timestamp:
        continue
    transmit_lines.append((sender, recver, start_timestamp, end_timestamp))

# with open("transmit_profiling.csv", "w", encoding="utf-8") as f:
#     for (sender_rank, receiver_rank) in sorted(transmtit_profiling.keys()):
#         f.write(f"From {rank__to__acc_name.get(sender_rank, f'rank{sender_rank}')}(rank[{sender_rank}]) "
#                 f"to {rank__to__acc_name.get(receiver_rank, f'rank{receiver_rank}')}(rank[{receiver_rank}])\n")
#         f.write("size(byte),avg time(ns)\n")
#         size__to__sumcnt = transmtit_profiling[(sender_rank, receiver_rank)]
#         for size in sorted(size__to__sumcnt.keys()):
#             sum_time, cnt = size__to__sumcnt[size]
#             avg_time = sum_time // max(1, cnt)
#             f.write(f"{size},{avg_time}\n")




import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



plt.figure(figsize=(width, high))

for (rank, start_ts, end_ts), color in zip(op_lines, op_lines_color):
    start_ts = start_ts / 1000000000
    end_ts = end_ts / 1000000000

    if end_ts - start_ts < 0.001:
        end_ts += 0.01 # 如果算子间隔太短，这里稍微加一点点，否则图里体现不出来
    plt.plot([start_ts, end_ts], [rank, rank], color=color, solid_capstyle='butt', linewidth=20)

for sender, recver, start_ts, end_ts in transmit_lines:
    start_ts = start_ts / 1000000000
    end_ts = end_ts / 1000000000
    plt.plot([start_ts, end_ts], [sender, recver], color="black", linewidth=1)

ranks = sorted(rank__to__acc_name.keys())

plt.yticks(ticks=ranks, labels=[f"Rank[{rank}]" for rank in ranks])

if len(ranks) > 0:
    ymin = min(ranks)
    ymax = max(ranks)
    plt.ylim(ymin - 1, ymax + 1)

plt.xlim(left = 1)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5)) # 每10一个刻度

plt.xlabel('Time (s)')

plt.savefig(log_filename + ".pdf", bbox_inches='tight')
