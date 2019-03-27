import sys 
import os 
import subprocess as sb

bashcmd = []
with open("setup.txt", 'r') as f:
    while True:
        line = f.readline()
        bashcmd.append(line)
        if not line:
            break

for cmd in bashcmd: 
    process = sb.Popen(cmd.split(), stdout=sb.PIPE)
    output, error = process.communicate()
