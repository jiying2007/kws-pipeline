"""Stdlib pipe capture with a strict on-disk prefix limit and bounded reads.

The caller owns process cleanup. Call ``finish`` after stopping/reaping writers;
it drains the remaining pipe bytes without an unbounded wait, then closes only
its input pipe and selector. The output stream remains the caller's property.
"""
import os
import selectors
import time


class BoundedLog:
    CHUNK_BYTES = 65536
    PUMP_BYTES = 4 * CHUNK_BYTES

    def __init__(self, pipe, stream, limit):
        if not isinstance(limit, int) or limit < 0:
            raise ValueError("Log limit must be a nonnegative integer")
        self.pipe = pipe
        self.stream = stream
        self.limit = limit
        self.written = 0
        self.observed_bytes = 0
        self.truncated = False
        self.eof = False
        self.drain_timed_out = False
        self.closed = False
        self.selector = selectors.DefaultSelector()
        try:
            os.set_blocking(pipe.fileno(), False)
            self.selector.register(pipe, selectors.EVENT_READ)
        except BaseException:
            self.selector.close()
            pipe.close()
            raise

    def pump(self, timeout=0.2):
        """Read at most PUMP_BYTES; a busy writer cannot starve watchdog checks."""
        if self.closed or self.eof:
            return 0
        if not self.selector.select(max(0, timeout)):
            return 0
        consumed = 0
        while consumed < self.PUMP_BYTES:
            try:
                chunk = os.read(self.pipe.fileno(), min(self.CHUNK_BYTES, self.PUMP_BYTES - consumed))
            except BlockingIOError:
                break
            if not chunk:
                self.eof = True
                self.selector.unregister(self.pipe)
                break
            consumed += len(chunk)
            self.observed_bytes += len(chunk)
            keep = min(len(chunk), self.limit - self.written)
            if keep:
                self.stream.write(chunk[:keep])
                self.written += keep
            if keep < len(chunk):
                self.truncated = True
        self.stream.flush()
        return consumed

    def finish(self, timeout=1.0):
        """Drain after process cleanup, then close even if a writer retains the pipe."""
        if self.closed:
            return
        deadline = time.monotonic() + max(0, timeout)
        try:
            while not self.eof:
                self.pump(timeout=min(0.05, max(0, deadline - time.monotonic())))
                if not self.eof and time.monotonic() >= deadline:
                    self.drain_timed_out = True
                    break
        finally:
            self.close()

    def close(self):
        if not self.closed:
            self.selector.close()
            self.pipe.close()
            self.closed = True
