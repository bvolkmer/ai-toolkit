from typing import override
from tqdm import tqdm
from pathlib import Path
import json


class ToolkitProgressBar(tqdm):
    def __init__(self, progress_file: Path | None = None, *args, **kwargs):
        self.progress_file = (
            progress_file.open("a", buffering=1) if progress_file is not None else None
        )
        super().__init__(*args, **kwargs)
        self.paused = False
        self.last_time = self._time()

    def pause(self):
        if not self.paused:
            self.paused = True
            self.last_time = self._time()

    def unpause(self):
        if self.paused:
            self.paused = False
            cur_t = self._time()
            self.start_t += cur_t - self.last_time
            self.last_print_t = cur_t

    def update(self, *args, **kwargs):
        if not self.paused:
            super().update(*args, **kwargs)

    def refresh(self, *args, **kwargs):
        if self.progress_file is not None:
            self.progress_file.write(json.dumps(self.format_dict) + "\n")

        super().refresh(*args, **kwargs)

    def close(self):
        if self.progress_file is not None:
            self.progress_file.write(json.dumps(self.format_dict) + "\n")
            self.progress_file.close()

        super().close()
