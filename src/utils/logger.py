from pathlib import Path
import csv
import json


class Logger:

    def __init__(
        self,
        output_dir,
    ):
        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.log_file = (
            self.output_dir
            / "metrics.csv"
        )

        self._initialized = False

    def log(
        self,
        step,
        metrics,
    ):
        row = {
            "step": step,
            **metrics,
        }

        print(
            " | ".join(
                f"{key}={value:.6f}"
                if isinstance(
                    value,
                    float,
                )
                else f"{key}={value}"
                for key, value
                in row.items()
            )
        )

        write_header = (
            not self.log_file.exists()
        )

        with open(
            self.log_file,
            "a",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=row.keys(),
            )

            if write_header:
                writer.writeheader()

            writer.writerow(row)

    def save_config(self, config):

        path = (
            self.output_dir
            / "config.json"
        )

        with open(
            path,
            "w",
        ) as f:

            json.dump(
                config,
                f,
                indent=4,
            )