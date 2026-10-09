# keepawake.py — prevents system sleep for N minutes (wall-clock), for long training runs.
# Usage: python keepawake.py <minutes>
# Note: does NOT keep the display on and does NOT defeat lid-close sleep settings.
import ctypes, sys, time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
k = ctypes.windll.kernel32


def hold():
    k.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


hold()
print(f"keepawake: holding system awake for {minutes:g} minutes", flush=True)
end = time.time() + minutes * 60
while time.time() < end:
    time.sleep(60)
    hold()
k.SetThreadExecutionState(ES_CONTINUOUS)  # release
print("keepawake: done, released after", minutes, "minutes")
