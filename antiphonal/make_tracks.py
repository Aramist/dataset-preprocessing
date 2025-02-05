import json
from pathlib import Path


def make_sleap_command(vid_file_path: Path, sleap_model_path: Path, output_path: Path):
    track_command = (
        "sleap-track "
        f'--model "{sleap_model_path}" '
        f"--output {output_path} "
        "--gpu auto "
        f"{vid_file_path}"
    )
    converted_path = output_path.with_suffix(".analysis.h5")
    cvt_command = (
        "sleap-convert " "--format analysis " f"-o {converted_path} " f"{output_path}"
    )
    if output_path.exists() and not converted_path.exists():
        return cvt_command
    elif output_path.exists():
        return None
    else:
        return f"{track_command}; {cvt_command}"


def wrap_command_environment(cmd: str, log_path=None):
    # Wrap the command in a script that sets the environment
    cmd = f'source /mnt/home/atanelus/.bashrc; eval "$(/mnt/home/atanelus/miniconda3/bin/conda shell.bash hook)"; conda activate sleap; {cmd}'
    if log_path is not None:
        cmd = f"( {cmd} ) &> {log_path}"
    return cmd


def main():
    with open("consts.json", "r") as ctx:
        consts = json.load(ctx)
    data_dirs = map(Path, consts["video_alias_dirs"])

    avi_files = []
    for data_dir in data_dirs:
        avi_files.extend(data_dir.glob("*.avi"))
        avi_files.extend(data_dir.glob("*.mp4"))

    # attempt to run sleap
    working_dir = Path(consts["working_dir"])
    trained_model_paths = list(map(Path, consts["sleap_model_dirs"]))
    path_for_model_idxs = consts["model_for_video"]
    track_dir = working_dir / consts["sleap_track_dir"]
    track_dir.mkdir(exist_ok=True, parents=True)

    log_dir = working_dir / Path("sleap_tracking_logs")
    log_dir.mkdir(exist_ok=True, parents=True)

    for avi_file in sorted(avi_files):
        fdate = "_".join(avi_file.stem.split("_")[:7])
        new_name = fdate + ".tracks.h5.slp"
        output_path = track_dir / new_name
        log_path = log_dir / (fdate + ".log")
        model_path_idx = path_for_model_idxs.get(avi_file.name, 0)
        model_path = trained_model_paths[model_path_idx]

        if output_path.exists():
            continue

        cmd = make_sleap_command(avi_file, model_path, output_path)
        if not cmd:
            continue
        cmd = wrap_command_environment(cmd, log_path)
        print(cmd)


if __name__ == "__main__":
    main()
