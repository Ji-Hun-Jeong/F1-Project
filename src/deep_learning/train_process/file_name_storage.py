import random


class FileNameStorage:
    def __init__(self):
        self.file_name_list: list[str] = []

    def add_file_name(self, file_name: str) -> None:
        self.file_name_list.append(file_name)

    def print_all_file_name(self) -> None:
        for file_name in self.file_name_list:
            print(file_name)

    def get_random_file_name(self) -> str:
        idx = random.randint(0, len(self.file_name_list) - 1)
        return self.file_name_list[idx]