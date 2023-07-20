import os, io
import git
from contextlib import contextmanager
import sys

def get_git_root():
    """
    Return the path to the current git directory's root.

    output git_root (str): Path of the current git directory's root.
    """
    git_repo = git.Repo(os.path.abspath(__file__), search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

@contextmanager
def nostd():
    """
    Decorator supressing console output. Source : https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
    """
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = io.BytesIO()
    sys.stderr = io.BytesIO()
    yield
    sys.stdout = save_stdout
    sys.stderr = save_stderr

def updateFileCount(path):
    """
    :param1 path (string): Path to the file counting the number of experimentation output I already generated.

    :output updated (int): New number of output file created after the current experimentation.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            value = int(f.read())
        updated = value + 1
    else:
        updated = 1
    with open(path, "w") as f:
        f.write(str(updated))    
    return updated

def readFileCount(path):
    """
    :param1 path (string): Path to the file counting the number of experimentation output I already generated.

    :output updated (int): New number of output file created after the current experimentation.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            value = int(f.read())
    return value