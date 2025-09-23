from datetime import datetime
import json
import os
from transformers import TrainerCallback


class LogToFileCallback(TrainerCallback):
    def __init__(self, log_dir):
        current_timestamp = datetime.now().timestamp()
        log_filename = f'train_log.{current_timestamp}.log' 
        self.log_file = os.path.join(log_dir, log_filename)
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        # Extract directory path from log_file
        log_dir = os.path.dirname(self.log_file)
        # Create the directory if it does not exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Ensure the directory exists before opening the log file
        self._ensure_log_directory()

        logs = logs.copy()
        logs["epoch"] = state.epoch
        with open(self.log_file, "a") as f:
            f.write(json.dumps(logs) + "\n")