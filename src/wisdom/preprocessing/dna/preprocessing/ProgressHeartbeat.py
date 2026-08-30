"""Occasional parent-process liveness logs for long LambdaForge maps."""

import time
import lambdaforge as lf

from threading import Event, Thread


class ProgressHeartbeat:
    """Emit race-free liveness messages without duplicating LambdaForge progress counts."""

    def __init__(
        self,
        work            : lf.Work,
        phase           : str,
        interval_seconds: float,
    ) -> None:
        """Configure one parent-only heartbeat.

        Args:
            work: Active LambdaForge Work whose thread-safe logger receives each line.
            phase: Human-readable operation shown in the log.
            interval_seconds: Seconds between liveness messages.
        """
        self.work             = work
        self.phase            = phase
        self.interval_seconds = interval_seconds
        self.started          = 0.0
        self.stop             = Event()
        self.thread           = Thread(target=self.run, daemon=True)

    def __enter__(self) -> "ProgressHeartbeat":
        """Start the background heartbeat.

        Returns:
            This heartbeat, ready to be stopped by the context-manager exit.
        """
        self.started = time.monotonic()
        self.thread.start()
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        """Stop the heartbeat after success or failure.

        Args:
            exception_type: Exception class raised inside the guarded map, when present.
            exception: Exception instance raised inside the guarded map, when present.
            traceback: Exception traceback raised inside the guarded map, when present.
        """
        self.stop.set()
        self.thread.join()

    def run(self) -> None:
        """Log elapsed time until the parent marks the map complete."""
        while not self.stop.wait(self.interval_seconds):
            elapsed_minutes = (time.monotonic() - self.started) / 60.0
            self.work.log(
                f"{self.phase} remains active after {elapsed_minutes:.1f} min; "
                "lf top contains the exact completed/total count"
            )
