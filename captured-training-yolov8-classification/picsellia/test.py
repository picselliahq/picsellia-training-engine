import time

t = 0

while True:
    print(f"ok {t}")
    time.sleep(1)
    t += 1

    if t == 60:
        raise RuntimeError("rip")
