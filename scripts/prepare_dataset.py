"""
Adapted from https://github.com/pacman100/DHS-LLM-Workshop/tree/main/personal_copilot/dataset_generation
"""
import os
import pandas as pd
from nbformat import reads, NO_CONVERT
from tqdm import tqdm
from typing import Dict
import subprocess

# Block the following formats.
IMAGE = ["png", "jpg", "jpeg", "gif"]
VIDEO = ["mp4", "jfif"]
DOC = [
    "key",
    "PDF",
    "pdf",
    "docx",
    "xlsx",
    "pptx",
]
AUDIO = ["flac", "ogg", "mid", "webm", "wav", "mp3"]
ARCHIVE = ["jar", "aar", "gz", "zip", "bz2"]
MODEL = ["onnx", "pickle", "model", "neuron"]
OTHERS = [
    "npy",
    "index",
    "inv",
    "index",
    "DS_Store",
    "rdb",
    "pack",
    "idx",
    "glb",
    "gltf",
    "len",
    "otf",
    "unitypackage",
    "ttf",
    "xz",
    "pcm",
    "opus",
]
ANTI_FOMATS = tuple(IMAGE + VIDEO + DOC + AUDIO + ARCHIVE + OTHERS)

# Block the following paths.
BLOCK_PATH = [".git", "__pycache__", "xcodeproj", ".vscode"]

def get_repo(username, repository):
    """Clones a repository."""
    if os.path.isdir(repository):
        subprocess.run(["git", "pull"], cwd=repository)
        return

    auth_name = os.getenv('GH_USER')
    auth_token = os.getenv('GH_TOKEN')
    if auth_name is None:
        repo_url = f"https://github.com/{username}/{repository}.git"
    else:
        repo_url = f"https://{auth_name}:{auth_token}@github.com/{username}/{repository}.git"

    subprocess.run(["git", "clone", repo_url, repository])

def filter_code_cell(cell) -> bool:
    """Filters a code cell w.r.t shell commands, etc."""
    only_shell = cell["source"].startswith("!")
    only_magic = "%%capture" in cell["source"]
    if only_shell or only_magic:
        return False
    else:
        return True


def process_file(file_path: str) -> str:
    """Processes a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            if file_path.endswith("ipynb"):
                code_cell_str = ""
                notebook = reads(content, NO_CONVERT)

                code_cells = [
                    c
                    for c in notebook["cells"]
                    if c["cell_type"] == "code"
                    if filter_code_cell(c)
                ]

                for cell in code_cells:
                    code_cell_str += cell["source"]
                content = code_cell_str
    except Exception:
        content = ""

    return content


def read_repository_files(directory) -> pd.DataFrame:
    """Reads the files from a locally cloned repository."""
    file_paths = []
    df = pd.DataFrame(columns=["repo_id", "file_path", "content"])
    chunk_flag = 0

    # Recursively find all files within the directory
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not file_path.endswith(ANTI_FOMATS) and all(
                k not in file_path for k in BLOCK_PATH
            ):
                file_paths.append(file_path)

    # Process files sequentially.
    print(f"Total file paths: {len(file_paths)}.")
    print("Reading file contents...")

    for i, file_path in enumerate(tqdm(file_paths)):
        file_content = process_file(file_path)

        if file_content != "":
            temp_df = pd.DataFrame.from_dict(
                {
                    "repo_id": [directory], 
                    "file_path": [file_path], 
                    "content": [file_content]
                    })
            df = pd.concat([df, temp_df], ignore_index=True)

    return df

def create_dataset_from_git_repo(username, repository) -> str:
    """Create a dataset from source code in a GitHub repository."""
    get_repo(username, repository)
    data = read_repository_files(repository)
    parquet_file=f"{username}_{repository}.parquet"
    data.to_parquet(parquet_file)
    return parquet_file
    