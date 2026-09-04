import contextlib
import json
import subprocess
import sys

import psutil

from scripts.profile_serving import stop_process_tree


def test_profile_cleanup_reaches_stubborn_children_in_new_sessions():
    child = """
import signal,time
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print('ready',flush=True)
time.sleep(300)
"""
    parent = f"""
import signal,subprocess,sys,time
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGCHLD, signal.SIG_IGN)
child=subprocess.Popen([sys.executable,'-c',{json.dumps(child)}],
    start_new_session=True,stdout=subprocess.PIPE,text=True)
assert child.stdout.readline().strip()=='ready'
print(child.pid,flush=True)
time.sleep(300)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", parent],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    child_pid = int(process.stdout.readline())
    try:
        assert psutil.Process(child_pid).is_running()
        stop_process_tree(process, grace_seconds=0.2)
        assert process.poll() is not None
        assert not psutil.pid_exists(child_pid)
    finally:
        for pid in (child_pid, process.pid):
            with contextlib.suppress(psutil.NoSuchProcess):
                psutil.Process(pid).kill()
        process.wait(timeout=5)
