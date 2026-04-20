import time
import sys

class ProgressTracker:
    """
    A simple progress indicator for long-running tasks.
    """
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()

    def update(self, steps: int = 1, message: str = ""):
        self.current_step += steps
        percentage = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time

        # Simple progress bar
        bar_length = 20
        filled_length = int(bar_length * self.current_step // self.total_steps)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        sys.stdout.write(f"\r{self.description}: |{bar}| {percentage:.1f}% ({self.current_step}/{self.total_steps}) {message}")
        sys.stdout.flush()

        if self.current_step >= self.total_steps:
            sys.stdout.write(f"\nCompleted in {elapsed:.2f}s\n")
            sys.stdout.flush()

    @classmethod
    def track_iterable(cls, iterable, description="Processing"):
        total = len(iterable)
        tracker = cls(total, description)
        for i, item in enumerate(iterable):
            yield item
            tracker.update(1)
