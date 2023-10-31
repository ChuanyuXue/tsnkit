#!/bin/bash

current_pid=$$
pids=$(pgrep -f "killif.sh")

for pid in $pids; do
    if [ $pid != $current_pid ]; then
        kill $pid
    fi
done
  
while true; do
    for pid in $(ps -eo pid,comm --sort=-%mem | awk '{if ($2 ~ /python|cpoptimizer/) print $1}'); do
        mem=$(grep VmRSS /proc/$pid/status | awk '{print $2}')
        uptime=$(ps -p $pid -o etimes | grep -Eo '[0-9]+')
        if [ $pid = $2 ]; then
            continue
        fi
        if [ $mem -gt $1 ] || [ $uptime -gt 5400 ]; then
            echo "Killing $pid, Memory: $mem"
            kill -2 $pid
            sleep 0.1
        fi
    done
    sleep 1
done                                                                     