#!/usr/bin/env bash
# Log GPU power, temperature, clocks and memory every 5 s to runs/logs/gpu_dmon.log while GPU jobs run, so a crash
# (the PRO 6000 has fallen off the bus twice: Xid 79) leaves a record of the conditions just before it.
# Idempotent: starts one logger, does nothing if it is already running.
PID=runs/logs/gpu_dmon.pid
mkdir -p runs/logs
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then exit 0; fi
nohup nvidia-smi dmon -s pucvmet -d 5 -o DT >> runs/logs/gpu_dmon.log 2>&1 &
echo $! > "$PID"
