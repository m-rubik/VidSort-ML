def get_unique_filename(path):
    max_num = 0
    for existing_file in path.iterdir():
        file_number = int(existing_file.stem)
        max_num = file_number if file_number > max_num else max_num
    return str(max_num+1)