import os
import time

from tensorflow.keras.callbacks import TensorBoard as tb


def main():
    tb.configure(bind_all=True, logdir="logs\\fit")
    url = tb.launch()
    print("TensorBoard %s started at %s" % (tensorboard.__version__, url))
    pid = os.getpid()
    print("PID = %d; use 'kill %d' to quit" % (pid, pid))
    while True:
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            break
    print()
    print("Shutting down")


if __name__ == "__main__":
    main()