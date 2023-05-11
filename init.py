import git
import os
import gdown

def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

def downloadRequirements(base_path):
    names = ["serialized_w2v.pkl",
             "GoogleNews-vectors-negative300.bin.gz"]
    urls = ["https://drive.google.com/file/d/1nGMHQT8ULVwa-18s6mPBWchWyd13AJAe/view?usp=share_link",
            "https://drive.google.com/file/d/1pUaT_XDjFvoaSQsYCyGAyqcUL8Nu_nVG/view?usp=share_link"]

    for url, name in zip(urls, names):
        gdown.download(url, os.path.join(base_path, name), quiet=False, fuzzy=True)

#finding and creating missing directories
repository_path = get_git_root(os.path.abspath(__file__))
ressources_path = input("Absolute path to download ressources: ")
ressources_path = ressources_path.replace("\\", "/")

if not os.path.exists(ressources_path):
    os.makedirs(ressources_path)

paths = [repository_path, ressources_path]

with open(os.path.join(repository_path, "myPaths.txt"), "w") as f:
    for path in paths:
        f.write(path + "\n")

#loading required data

downloadRequirements(ressources_path)

print("Packages ready to be used")