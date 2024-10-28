import json
from pathlib import Path
from pprint import pprint

from rclone_python import rclone
from tqdm import tqdm

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

GDRIVE_PATH = Path(consts["gdrive_path"])
DOWNLOAD_PATH = Path(consts["download_path"])


def download_data():
    """Downloads the dataset from google drive"""
    sessions = rclone.ls(str(GDRIVE_PATH), max_depth=1, args=["--drive-shared-with-me"])
    sessions = map(
        lambda meta: meta["Path"], filter(lambda meta: meta["IsDir"], sessions)
    )
    sessions = list(sessions)  # list of strings
    print("Copying the following sessions:")
    pprint(sessions)

    for session in tqdm(sessions):
        (DOWNLOAD_PATH / session).mkdir(exist_ok=True)
        session_path = GDRIVE_PATH / session
        files_in_session = rclone.ls(
            str(session_path), max_depth=1, args=["--drive-shared-with-me"]
        )
        files_in_session = map(
            lambda meta: meta["Path"],
            filter(lambda meta: not meta["IsDir"], files_in_session),
        )
        files_in_session = list(files_in_session)
        for file in files_in_session:
            file_path_remote = session_path / file
            file_path_local = DOWNLOAD_PATH / session / file
            if file_path_local.exists():
                continue
            rclone.copy(
                str(file_path_remote),
                str(file_path_local),
                args=["--drive-shared-with-me"],
            )


if __name__ == "__main__":
    DOWNLOAD_PATH.mkdir(exist_ok=True, parents=True)
    download_data()
