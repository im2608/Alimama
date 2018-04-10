'''
Created on Mar 1, 2018

@author: Heng.Zhang
'''

import subprocess  
import time

runningSubProcesses = {}


def submiteOneSubProcess(date, dffilename):
    cmdLine = "python preprocess.py d=%s f=%s" % (date, dffilename)
    sub = subprocess.Popen(cmdLine, shell=True)
    runningSubProcesses[(date, time.time(), dffilename)] = sub
    print("running cmd line: %s" % cmdLine)
    time.sleep(1)
    return


def waitSubprocesses():
    for start_end_date_str in runningSubProcesses:
        sub = runningSubProcesses[start_end_date_str]
        ret = subprocess.Popen.poll(sub)
        if ret == 0:
            runningSubProcesses.pop(start_end_date_str)
            return start_end_date_str
        elif ret is None:
            time.sleep(1) # running
        else:
            runningSubProcesses.pop(start_end_date_str)
            return start_end_date_str
    return (0, 0)

def main():
    train_date = ['2018-09-17', '2018-09-18', '2018-09-19', '2018-09-20', '2018-09-21', '2018-09-22', '2018-09-23', '2018-09-24']
    for each in train_date:
        submiteOneSubProcess(each, "train")
        
    test_date = ['2018-09-24', '2018-09-25']
    for each in test_date:
        submiteOneSubProcess(each, "test")

    while True:
        start_end_date_str = waitSubprocesses()
        if ((start_end_date_str[0] != 0 and start_end_date_str[1] != 0)):
                    print("after waitSubprocesses, subprocess [%s, %s] finished, took %d seconds, runningSubProcesses len is %d" % 
                          (start_end_date_str[0], start_end_date_str[2], time.time() - start_end_date_str[1], len(runningSubProcesses)))
        if (len(runningSubProcesses) == 0):
            break

    return

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

if __name__ == '__main__':
    main()