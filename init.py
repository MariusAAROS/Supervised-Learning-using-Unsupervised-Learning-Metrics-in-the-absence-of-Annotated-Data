import git
import os

def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

repository_path = get_git_root(os.path.abspath(__file__))
ressources_path = input("Absolute path to download ressources: ")
ressources_path = ressources_path.replace("\\", "/")

if not os.path.exists(ressources_path):
    os.makedirs(ressources_path)

paths = [repository_path, ressources_path]

with open(os.path.join(repository_path, "myPaths.txt"), "w") as f:
    for path in paths:
        f.write(path + "\n")
print("Packages ready to be used")