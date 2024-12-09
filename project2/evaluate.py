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
import dotenv
import glob
import httpx
import json
import os
import pandas as pd
import shutil
import sys
from datetime import datetime, timedelta, timezone
from collections import namedtuple, Counter
from platformdirs import user_data_dir
from rich.console import Console
from subprocess import run, PIPE

# Deadline for repo is 12 Dec 2024 EOD AOE. If you're hacking dates, remember:
# 1. Change your commit time to before the deadline
# 2. We'll clone at some unknown time after this deadline. Time it before that.
deadline = datetime(2024, 12, 12, 23, 59, 59, tzinfo=timezone(timedelta(hours=-12)))

console = Console()
root = user_data_dir("tds-sep-24-project-2", "tds")
HEAD = namedtuple("HEAD", ["owner", "repo", "branch"])
Eval = namedtuple("Eval", ["marks", "total", "test", "reason"])

# Faculty evaluation of code and output will use LLM Foundry. Student evaluation will use AIProxy.
dotenv.load_dotenv()
if "LLMFOUNDRY_TOKEN" in os.environ:
    headers = {"Authorization": f"Bearer {os.environ['LLMFOUNDRY_TOKEN']}:tds-sep-2024-project-2"}
    openai_api_base = "https://llmfoundry.straive.com/openai/v1"
elif "AIPROXY_TOKEN" in os.environ:
    headers = {"Authorization": f"Bearer {os.environ['AIPROXY_TOKEN']}"}
    openai_api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
else:
    raise ValueError("Missing AIPROXY_TOKEN")


# Sample datasets from https://drive.google.com/drive/folders/1KNGfcgA1l2uTnqaldaX6LFr9G1RJQNK3
sample_datasets = {
    "goodreads.csv": "1oYI_Vdo-Xmelq7FQVCweTQgs_Ii3gEL6",
    "happiness.csv": "15nasMs0VKVB4Tm7-2EKYNpDzPdkiRW1q",
    "media.csv": "10LcR2p6SjD3pWASVp5M4k94zjMZD6x5W",
}


def log(msg: str, last=False):
    """Log a message to the console."""
    console.print(" " * 200, end="\r")
    console.print(msg, **({} if last else {"end": "\r"}))


def download_datasets():
    """Download the datasets from Google Drive."""
    datasets_dir = os.path.join(root, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    for name, id in sample_datasets.items():
        target = os.path.join(datasets_dir, name)
        if not os.path.exists(target) or os.path.getsize(target) == 0:
            url = f"https://drive.usercontent.google.com/download?id={id}"
            log(f"[yellow]DOWNLOAD[/yellow] {name}...")
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


def clone_latest_branch(id: str, head: HEAD, deadline: datetime):
    """Ensure the latest commit on the branch is before the deadline."""
    repo_path = os.path.join(root, id)

    # Clean up the repo if it exists to avoid problems with forced pushes.
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    # Clone the repo
    repo_url = f"https://github.com/{head.owner}/{head.repo}.git"
    cmd = ["git", "clone", "-q", "--single-branch", "-b", head.branch, repo_url, repo_path]
    log(f"[blue]{id}[/blue] [yellow]CLONE[/yellow] {repo_url}")
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    run(cmd, check=True, capture_output=True, text=True, env=env)

    # Get the latest commit before the deadline
    log_cmd = ["git", "-C", repo_path, "log", "-q", head.branch]
    log_cmd.extend(["--before", deadline.isoformat(), "--format=%H", "-n", "1", "--"])
    log(f"[blue]{id}[/blue] [yellow]LOG[/yellow] {repo_url}")
    commit = run(log_cmd, stdout=PIPE, text=True, check=True).stdout.strip()

    # Checkout the commit
    if commit:
        log(f"[blue]{id}[/blue] [yellow]CHECKOUT[/yellow] {commit}")
        run(["git", "-C", repo_path, "checkout", "--quiet", commit], check=True)
    else:
        raise ValueError(f"No commits on branch {head.branch} before {deadline}")


def has_mit_license(id: str, evals: list[Eval]) -> bool:
    """Check if root/{id}/LICENSE is an MIT license."""
    with open(os.path.join(root, id, "LICENSE")) as f:
        marks = 1.0 if "permission is hereby granted, free of charge" in f.read().lower() else 0.0
        evals.append(Eval(marks, 1.0, "mit_license", "present" if marks else "missing"))


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
        evals.append(Eval(marks, total, pattern, "present" if marks else "missing"))


def run_on_dataset(id: str, dataset: str, evals: list[Eval]):
    msg = f"[blue]{id}[/blue] [yellow]uv run autolysis[/yellow] {dataset}"
    cwd = os.path.join(root, id, "eval", dataset)
    os.makedirs(cwd, exist_ok=True)
    script = os.path.join(root, id, "autolysis.py")
    cmd = ["uv", "run", script, os.path.join(root, "datasets", dataset)]
    log(msg)
    result = run(cmd, check=False, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        evals.append(Eval(0.0, 0.5, msg, result.stderr))
        log(f"{msg} [red]FAIL[/red]: {result.stderr}", last=True)
        return False
    else:
        evals.append(Eval(0.5, 0.5, f"uv run autolysis {dataset}", "ran"))
        return True


# Define the code quality evaluation attributes. WARNING: This is work in progress.
Qual = namedtuple("Qual", ["group", "name", "description"])
code_system = """
You are an expert Python programmer. Evaluate the quality of this Python code. Respond as JSON.
For each code quality attribute:
- FIRST explain your reasoning, citing code blocks that provide evidence for and against the attribute.
- THEN answer as a boolean. Use your judgement critically using your reasoning. Prefer false if unsure.
"""

# The criteria are mentioned in the project description. But the exact prompts are a secret.
# But here are a few examples:
# 1|uses_functions|Is the code broken down into functions APTLY, to avoid code duplication, reduce complexity, and improve re-use?
# 1|consistent_coding_style|Is the ENTIRE code written in a consistent style?
# 5|concise_prompts|Are prompts concise, entity-dense, and to the point, rather than unnecessarily verbose or repetitive?
code_quality = []
for line in os.environ.get("CODE_QUALITY", "").split("\n"):
    if not line.strip() or line.startswith("#"):
        continue
    group, name, description = line.split("|", 2)
    code_quality.append(Qual(group, name, description))

# Attributes belong to one of 7 groups (as defined in the project description). Count them.
code_quality_group_counts = Counter(attribute.group for attribute in code_quality)

# Define the JSON schema for the code quality evaluation. {attribute.name: {reasoning, answer}}
# Put reasoning first to avoid rationalization bias.
code_quality_properties = {
    attribute.name: {
        "type": "object",
        "description": attribute.description,
        "properties": {"reasoning": {"type": "string"}, "answer": {"type": "boolean"}},
        "required": ["reasoning", "answer"],
        "additionalProperties": False,
    }
    for attribute in code_quality
}
code_quality_schema = {
    "type": "object",
    "properties": code_quality_properties,
    "required": list(code_quality_properties.keys()),
    "additionalProperties": False,
}
json_schema = {"name": "quality", "strict": True, "schema": code_quality_schema}


def evaluate_code_quality(id: str, evals: list[Eval]):
    # Students won't have access to code quality criteria (for now). Skip if so
    if not len(code_quality):
        return

    # Read the code
    with open(os.path.join(root, id, "autolysis.py")) as f:
        code = f.read()

    # Evaluate the code quality
    log(f"[blue]{id}[/blue] [yellow]CODE QUALITY[/yellow]")
    result = httpx.post(
        f"{openai_api_base}/chat/completions",
        headers=headers,
        json={
            "model": os.environ.get("MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": code_system},
                {"role": "user", "content": code},
            ],
            "response_format": {"type": "json_schema", "json_schema": json_schema},
        },
        timeout=60,
    )
    answers = json.loads(result.json()["choices"][0]["message"]["content"])
    for attribute in code_quality:
        total = 1.0 / code_quality_group_counts[attribute.group]
        ans = answers[attribute.name]
        evals.append(Eval(total if ans["answer"] else 0, total, attribute.name, ans["reasoning"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate student project submissions")
    parser.add_argument("url", nargs="*", help="GitHub raw URL for submission(s)")
    args = parser.parse_args()

    # If a URL is passed, get submission from that
    if len(args.url):
        submissions = []
        for url in args.url:
            head = parse_github_url(url)
            submissions.append([None, f"{head.owner}-{head.repo}", url])
        submissions = pd.DataFrame(submissions)
    # Else, the faculty will get all submissions from the Google Sheet and evaluate
    elif os.environ.get("SUBMISSION_URL"):
        submissions = pd.read_csv(os.environ["SUBMISSION_URL"])
    # Else, raise an error
    else:
        error = "[red]Missing URL[/red]: Usage `uv run project2.py https://raw.githubusercontent.com/...`"
        log(error, last=True)
        sys.exit(1)

    download_datasets()

    submissions["id"] = submissions[submissions.columns[1]].str.split("@").str[0]
    submissions["head"] = submissions[submissions.columns[2]].apply(parse_github_url)

    # Now, evalute each submission
    results = []
    for _, row in submissions.iterrows():
        evals = []
        try:
            clone_latest_branch(row.id, row["head"], deadline)
            has_mit_license(row.id, evals)
            has_required_files(row.id, evals)

            # Code evaluation
            evaluate_code_quality(row.id, evals)

            # Submission: Run evaluation for each sample dataset
            success = []
            for dataset in sample_datasets:
                success.append(run_on_dataset(row.id, dataset, evals))
            if all(success):
                evals.append(Eval(0.5, 0.5, "uv run autolysis *", "ran"))
            else:
                evals.append(Eval(0.0, 0.5, "uv run autolysis *", "failed"))

            result = pd.DataFrame(evals)
            result["id"] = row.id
            results.append(result)
            score, total = round(result.marks.sum(), 2), round(result.total.sum(), 2)
            log(f"[blue]{row.id}[/blue] [green]SCORE[/green] {score} / {total}", last=True)
        except Exception as e:
            log(f"[blue]{row.id}[/blue] [red]FAIL[/red] {e}", last=True)
            continue

    if len(results):
        results = pd.concat(results)
        results.to_csv(os.path.join(root, "results.csv"), index=False)
        print(results)
