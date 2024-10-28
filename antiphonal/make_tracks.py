import json
from pathlib import Path


def make_sleap_command(vid_file_path: Path, sleap_model_path: Path, output_path: Path):
    track_command = (
        "sleap-track "
        f"--model {sleap_model_path} "
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
    cmd = f'source /mnt/home/atanelus/.bashrc; eval "$(/mnt/home/atanelus/miniconda3/bin/conda shell.bash hook)"; conda activate sleap; module purge; module load modules/2.0-20220630; module load cuda/11.4.4; module load cudnn; {cmd}'
    if log_path is not None:
        cmd = f"( {cmd} ) &> {log_path}"
    return cmd


def main():
    with open("consts.json", "r") as ctx:
        consts = json.load(ctx)
    data_dir = Path(consts["recording_session_dir"])

    avi_files = data_dir.glob("*/*.avi")

    # attempt to run sleap
    working_dir = Path(consts["working_dir"])
    sleap_path = Path(consts["sleap_model_dir"])
    track_dir = working_dir / consts["sleap_track_dir"]
    track_dir.mkdir(exist_ok=True, parents=True)

    log_dir = working_dir / Path("sleap_tracking_logs")
    log_dir.mkdir(exist_ok=True, parents=True)

    for avi_file in sorted(avi_files):
        output_path = track_dir / (avi_file.parent.name + ".tracks.h5.slp")
        log_path = log_dir / (avi_file.parent.stem + ".log")
        cmd = make_sleap_command(avi_file, sleap_path, output_path)
        if not cmd:
            continue
        cmd = wrap_command_environment(cmd, log_path)
        print(cmd)


if __name__ == "__main__":
    main()
