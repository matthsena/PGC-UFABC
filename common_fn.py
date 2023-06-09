import os
from itertools import combinations

def all_combinations(itens):
    combinations_tuple = list(combinations(itens, 2))
    return combinations_tuple

def get_img_folders():
    folder_list = []
    for folder_name in os.listdir('img'):
        if os.path.isdir(os.path.join('img', folder_name)):
            folder_list.append(folder_name)
    return folder_list

def get_img_files(folder_path):
    file_list = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_list.append(file_name)
    return file_list