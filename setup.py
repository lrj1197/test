import sys 
import os 
import subprocess as sb

os.chdir("/")

bashCommand = "sudo pip install threaded"
process = sb.Popen(bashCommand.split(), stdout=sb.PIPE)
output, error = process.communicate()
# print(str(output).split("\n"))



