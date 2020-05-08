import os


def get_unique_filename(path):
    files = os.listdir(path)
    max_num = 0
    for existing_file in files:
        file_number = int(os.path.splitext(existing_file)[0])
        max_num = file_number if file_number > max_num else max_num
    return str(max_num+1)
