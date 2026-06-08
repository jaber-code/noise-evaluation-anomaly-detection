import os
from datetime import datetime
from typing import Optional


class DualLogger:
    """Writes lines to both stdout and an optional file.

    File layout when enabled:
        {output_dir}/{dataset_name}/{timestamp}_{training_type}_{model_type}/{prefix}_{timestamp}_{training_type}_{model_type}.txt
    """

    def __init__(
        self,
        enabled: bool,
        output_dir: str,
        dataset_name: str,
        training_type: str,
        model_type: str,
        prefix: str,
        run_id: Optional[str] = None,
    ):
        self.enabled = enabled
        self.filepath: Optional[str] = None

        if self.enabled:
            # Shared run_id so config + results files live in the same folder
            if run_id is None:
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            run_name = f"{training_type}_{model_type}_{run_id}"
            run_dir = os.path.join(output_dir, dataset_name, run_name)
            os.makedirs(run_dir, exist_ok=True)

            self.filepath = os.path.join(run_dir, f"{prefix}_{run_name}.txt")
            with open(self.filepath, "w"):
                pass

    def write(self, line: str = "") -> None:
        print(line)
        if self.enabled and self.filepath is not None:
            with open(self.filepath, "a") as f:
                f.write(line + "\n")