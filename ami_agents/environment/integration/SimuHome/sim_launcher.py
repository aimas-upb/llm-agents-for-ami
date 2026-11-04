"""Start the SimuHome simulator (its FastAPI app) under uvicorn on the loopback
interface, and say who terminates it.

On the shared dgxa100 nodes the simulator is shut down by a signal every few
minutes (103 of 107 simulator processes in one job; never the TD server started
by the same worker, never on the H200 nodes). uvicorn logs only "Shutting
down". This launcher blocks SIGTERM/SIGINT in every thread, receives them with
``sigwaitinfo`` in a watcher thread, logs the signal with the sender's pid, uid
and command line, then shuts the server down as uvicorn would have.

    python sim_launcher.py <port>        (cwd: the SimuHome checkout)
"""
import os
import signal
import sys
import threading

import uvicorn

SIGNALS = {signal.SIGTERM, signal.SIGINT}


def main() -> None:
    port = int(sys.argv[1])
    # Run as a script, Python puts this file's directory first on sys.path, not
    # the working directory; the app is imported from the SimuHome checkout (cwd).
    sys.path.insert(0, os.getcwd())
    signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)   # inherited by every thread
    server = uvicorn.Server(uvicorn.Config("src.simulator.api.app:app", host="127.0.0.1",
                                           port=port, reload=False))

    def watcher() -> None:
        info = signal.sigwaitinfo(SIGNALS)
        sender = "?"
        try:
            with open(f"/proc/{info.si_pid}/cmdline", "rb") as handle:
                sender = handle.read().replace(b"\0", b" ").decode(errors="replace")[:200]
        except OSError:
            pass
        print(f"SIGNAL {info.si_signo} received: si_code={info.si_code} si_pid={info.si_pid} "
              f"si_uid={info.si_uid} (own pid {os.getpid()}, parent {os.getppid()}) "
              f"sender_cmd={sender!r}", flush=True)
        server.handle_exit(info.si_signo, None)

    threading.Thread(target=watcher, daemon=True).start()
    server.run()


if __name__ == "__main__":
    main()
