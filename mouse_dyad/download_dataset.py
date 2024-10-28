import json
from pathlib import Path

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

GDRIVE_PATH = Path(consts["gdrive_path"])
DOWNLOAD_PATH = Path(consts["download_path"])


if __name__ == "__main__":
    DOWNLOAD_PATH.mkdir(exist_ok=True, parents=True)

    # there is only one folder, just print the command
    print(
        f"module load rclone; rclone copy {GDRIVE_PATH} {DOWNLOAD_PATH} --drive-shared-with-me --progress"
    )
