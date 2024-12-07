# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "python-dotenv",
#     "httpx",
#     "pandas",
#     "platformdirs",
#     "rich",
# ]
# ///

import argparse
import glob
import httpx
import os
import pandas as pd
import sys
import dotenv
from datetime import datetime, timedelta, timezone
from collections import namedtuple
from platformdirs import user_data_dir
from rich.console import Console
from subprocess import run, PIPE


dotenv.load_dotenv()
console = Console()
root = user_data_dir("tds-sep-24-project-2", "tds")
HEAD = namedtuple("HEAD", ["owner", "repo", "branch"])
Eval = namedtuple("Eval", ["marks", "total", "test"])

# Sample datasets from https://drive.google.com/drive/folders/1KNGfcgA1l2uTnqaldaX6LFr9G1RJQNK3
sample_datasets = {
    "goodreads.csv": "1oYI_Vdo-Xmelq7FQVCweTQgs_Ii3gEL6",
    "happiness.csv": "15nasMs0VKVB4Tm7-2EKYNpDzPdkiRW1q",
    "media.csv": "10LcR2p6SjD3pWASVp5M4k94zjMZD6x5W",
}


def log(msg: str, last=False):
    """Log a message to the console."""
    console.print(" " * 80, end="\r")
    console.print(msg, **({} if last else {"end": "\r"}))


def download_datasets():
    """Download the datasets from Google Drive."""
    datasets_dir = os.path.join(root, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    for name, id in sample_datasets.items():
        target = os.path.join(datasets_dir, name)
        if not os.path.exists(target) or os.path.getsize(target) == 0:
            log(f"Downloading {name}...")
            url = f"https://drive.usercontent.google.com/download?id={id}"
            result = httpx.get(url, timeout=30)
            result.raise_for_status()
            with open(target, "wb") as f:
                f.write(result.content)


def parse_github_url(raw_url: str) -> HEAD:
    """Parse the raw GitHub URL into an owner, repository, and branch."""
    parts = raw_url.split("/")
    # https://raw.githubusercontent.com/owner/repo/refs/heads/branch/autolysis.py
    if parts[5] == "refs" and parts[6] == "heads":
        return HEAD(parts[3], parts[4], parts[7])
    # https://raw.githubusercontent.com/owner/repo/branch/autolysis.py
    else:
        return HEAD(parts[3], parts[4], parts[5])


def clone_latest_branch(id: str, head: HEAD, deadline: datetime, reload=False):
    """Ensure the latest commit on the branch is before the deadline."""
    repo_path = os.path.join(root, id)
    if os.path.exists(repo_path) and not reload:
        return

    # Clone or fetch the repo
    repo_url = f"https://github.com/{head.owner}/{head.repo}.git"
    kwargs = {"check": True, "capture_output": True, "text": True, "env": {"GIT_TERMINAL_PROMPT": "0"}}
    if not os.path.exists(repo_path):
        cmd = ["git", "clone", "-q", "--single-branch", "-b", head.branch, repo_url, repo_path]
        run(cmd, **kwargs)
    else:
        run(["git", "-C", repo_path, "fetch", "--quiet", "origin", head.branch], **kwargs)
        run(["git", "-C", repo_path, "checkout", "--quiet", head.branch], **kwargs)

    # Get the latest commit before the deadline
    log_cmd = [
        "git",
        "-C",
        repo_path,
        "log",
        "-q",
        head.branch,
        "--before",
        deadline.isoformat(),
        "--format=%H",
        "-n",
        "1",
    ]
    commit = run(log_cmd, stdout=PIPE, text=True, check=True).stdout.strip()

    # Checkout the commit
    if commit:
        run(["git", "-C", repo_path, "checkout", "--quiet", commit], check=True)
    else:
        raise ValueError(f"No commits on branch {head.branch} before {deadline}")


def clone_all_submissions(submissions: pd.DataFrame, reload=False):
    """Clone all submissions from a spreadsheet with a timestamp, email, GitHub URL."""

    # Deadline for repo is 12 Dec 2024 EOD AOE. If you're hacking dates, remember:
    # 1. Change your commit time to before the deadline
    # 2. We'll clone at some unknown time after this deadline. Time it before that.
    deadline = datetime(2024, 12, 12, 23, 59, 59, tzinfo=timezone(timedelta(hours=-12)))

    submissions["id"] = submissions[submissions.columns[1]].str.split("@").str[0]
    submissions["head"] = submissions[submissions.columns[2]].apply(parse_github_url)
    for _, row in submissions.iterrows():
        head = row["head"]
        msg = f"{row.id}: {head.owner}/{head.repo}:{head.branch}"
        try:
            console.print(" " * 80, end="\r")
            console.print(f"[yellow]CLONE[/yellow] {msg}", end="\r")
            clone_latest_branch(row.id, head, deadline, reload)
        except Exception as e:
            console.print(f"[red]CLONE FAILED[/red] {msg}: {e}")
            continue


def has_mit_license(id: str, evals: list[Eval]) -> bool:
    """Check if root/{id}/LICENSE is an MIT license."""
    with open(os.path.join(root, id, "LICENSE")) as f:
        mark = 1.0 if "permission is hereby granted, free of charge" in f.read().lower() else 0.0
        evals.append(Eval(mark, 1.0, "1.0. Public repo with MIT LICENSE"))


def has_required_files(id: str, evals: list[Eval]):
    """Check if root/{id} has the required files."""
    required_files = {
        "autolysis.py": 0.4,
        "goodreads/README.md": 0.1,
        "goodreads/*.png": 0.1,
        "happiness/README.md": 0.1,
        "happiness/*.png": 0.1,
        "media/README.md": 0.1,
        "media/*.png": 0.1,
    }
    for index, (pattern, total) in enumerate(required_files.items()):
        marks = total if glob.glob(os.path.join(root, id, pattern)) else 0.0
        evals.append(Eval(marks, total, f"2.{index + 1}. Repo has {pattern}"))


def run_evaluation(id: str, dataset: str, evals: list[Eval]):
    log(f"{id}: [yellow]uv run autolysis[/yellow] {dataset}")
    cwd = os.path.join(root, id, "eval", dataset)
    os.makedirs(cwd, exist_ok=True)
    script = os.path.join(root, id, "autolysis.py")
    cmd = ["uv", "run", script, os.path.join(root, "datasets", dataset)]
    result = run(cmd, check=False, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        evals.append(Eval(0.0, 0.5, f"3. uv run autolysis {dataset} failed: {result.stderr}"))
        log(f"{id}: [red]uv run autolysis[/red] {dataset} failed: {result.stderr}", last=True)
    else:
        evals.append(Eval(0.5, 0.5, f"3. uv run autolysis {dataset} succeeded"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate student project submissions")
    parser.add_argument("url", nargs="?", help="GitHub raw URL for a single submission")
    parser.add_argument("--reload", action="store_true", help="Force reload repositories")
    args = parser.parse_args()

    # If a URL is passed, get submission from that
    if args.url:
        submissions = pd.DataFrame([[None, "me", args.url]])
    # Else, the faculty will get all submissions from the Google Sheet and evaluate
    elif os.environ.get("SUBMISSION_URL"):
        submissions = pd.read_csv(os.environ["SUBMISSION_URL"])
    # Else, raise an error
    else:
        log(
            "[red]Missing URL[/red]: Usage `uv run project2.py https://raw.githubusercontent.com/...`",
            last=True,
        )
        sys.exit(1)

    download_datasets()
    clone_all_submissions(submissions, args.reload)

    # Now, evalute each submission
    results = []
    for id in submissions["id"].tolist():
        evals = []
        log(f"Evaluating {id}...")
        try:
            log(f"[yellow]EVAL[/yellow] {id}")
            has_mit_license(id, evals)
            has_required_files(id, evals)
            for dataset in sample_datasets:
                run_evaluation(id, dataset, evals)
            evals.append(Eval(0.5, 0.5, "3. uv run autolysis * succeeded"))

            result = pd.DataFrame(evals)
            result["id"] = id
            results.append(result)
            log(
                f"[green]SCORE[/green] {id} {result.marks.sum()} / {result.total.sum()}", last=True
            )
        except Exception as e:
            log(f"[red]EVAL FAILED[/red] {e}", last=True)
            continue
