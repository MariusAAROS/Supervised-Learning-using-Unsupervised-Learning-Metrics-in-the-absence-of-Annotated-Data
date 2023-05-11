import git
import os
import requests



def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def downloadRequirements(pathToSave):
    urls = ["https://drive.google.com/file/d/1nGMHQT8ULVwa-18s6mPBWchWyd13AJAe/view?usp=share_link"]
    try:
        for url in urls:
            download_file_from_google_drive(url, pathToSave)
    except:
        return False
    finally:
        return True

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
downloadRequirements(os.path.join(ressources_path, "serialized_w2v.pkl"))

print("Packages ready to be used")