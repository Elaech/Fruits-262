import os


def filename_to_number(string_name):
    return float(string_name.split(".")[0])


def rename_with_numbers(path=None):
    print(path)
    files = [os.path.basename(file.path) for file in os.scandir(path)]
    files = sorted(files, key=filename_to_number)
    counter = 0
    for file in files:
        os.rename(os.path.join(path, file), os.path.join(path, f"{counter}.jpg"))
        counter += 1


def get_all_subfolders(path=None):
    if path:
        return [f.path for f in os.scandir(path) if f.is_dir()]
    return None


def mass_renumber(dir_path):
    fruit_paths = get_all_subfolders(dir_path)
    for path in fruit_paths:
        rename_with_numbers(path=path)


def renumber_one_label(path):
    rename_with_numbers(dir_path)


if __name__ == '__main__':
    # Example of renumbering jocote
    dir_path = "../HandFilteredDataset/jocote"
    renumber_one_label(dir_path)
