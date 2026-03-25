#!/bin/sh

if [ $# -lt 1 ]; then
    echo 'input pid'
    exit 1
fi

rm -f perf.*
perf record -F 99 -p $1 -g -o in-fb.data -- sleep 60 # 首先使用99HZ的采样频率，对pid为$1的进程进行采样，采样输出到in-fb.data中，采样时长为60秒
perf script -i in-fb.data &> perf.unfold
stackcollapse-perf.pl perf.unfold &> perf.folded
flamegraph.pl perf.folded > perf.svg
