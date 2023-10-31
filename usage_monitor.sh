#!/bin/bash

PATH=$PATH:/opt/homebrew/bin/sendemail
# Set the list of servers to monitor
SERVERS=(
  "cc@129.114.109.94"
  "cc@129.114.108.247"
  "cc@129.114.108.217"
  "cc@129.114.109.52"
  "cc@129.114.109.201"
  "cc@129.114.108.222"
  "cc@129.114.108.37"
  "cc@129.114.109.116"
)

# Set the email parameters
ADDRESS="skewcy@gmail.com"
PW="fswepkhikrspgzxm"
SUBJECT="CPU/Memory Usage on Multiple Servers"

# Loop through each server and get the CPU and memory usage
while true; do
  BODY="Please find the CPU and memory usage on the following servers:\n\n"
  COUNT=1
  for SERVER in "${SERVERS[@]}"
  do
    CPU_USAGE=$(ssh $SERVER "top -bn5 | grep \"Cpu(s)\" | tail -n 1 | awk '{print \$2 + \$4}'")
    # CPU_USAGE=$(ssh $SERVER "htop -C -n 1 | head -n 2 | tail -n 1 | awk '{print $2}'")
    MEM_USAGE=$(ssh $SERVER "free | grep Mem | awk '{print \$3/\$2 * 100.0}'")


    

    # Add the server usage information to the email body
    BODY+="Server: $SERVER AMD_ZEN_$COUNT \nCPU usage: $CPU_USAGE%\nMemory usage: $MEM_USAGE%\n\n"

    # Get the top-5 processes using the most CPU on the server
    PROCESSES=$(ssh $SERVER "ps aux --sort=-%cpu | head -6")

    # Add the top-5 processes information to the email body
    BODY+="Top-5 processes by CPU usage:\n$PROCESSES\n\n"

    COUNT=$((COUNT+1))
    echo "Retrieved usage information for $SERVER"
  done
  # Send the email notification with the usage information for all servers
  if [ -n "$BODY" ]; then
    sendemail -f "$ADDRESS" -t "$ADDRESS" -u "$SUBJECT" -m "$BODY" -s "smtp.gmail.com:587" -o tls=yes -xu "$ADDRESS" -xp "$PW"
  fi
  sleep 3600
done



