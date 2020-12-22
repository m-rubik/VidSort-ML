import pickle
import pprint
import os
import pathlib

DATABASES_ROOT_DIR = pathlib.Path(__file__).parents[1] / 'databases'

class database:

    name: str = "Default"
    data: dict

    def __init__(self, name):
        self.name = name
        self.data = {}

    def update(self, key, value):
        self.data[key] = value

    def display_entries(self):
        pprint.pprint(self.data)
        # for key, value in self.data.items():

    def search(self, names):
        if isinstance(names, str):
            names = [names]
        for movie, people in self.data.items():
            if all(name in people for name in names):
                names_str = " and ".join(names)
                print("Found", names_str, "in", movie+".")

def load_database(name):
    try:
        database_path = pathlib.Path(DATABASES_ROOT_DIR / name)
        with open(database_path, 'rb') as f:
            db = pickle.load(f)
    except FileNotFoundError:
        db = database(name)
    return db

def save_database(db):
    try:
        if not DATABASES_ROOT_DIR.is_dir():
            DATABASES_ROOT_DIR.mkdir(parents=True, exist_ok=False)
        database_path = pathlib.Path(DATABASES_ROOT_DIR / db.name)
        with open(database_path, 'wb') as f:
            pickle.dump(db, f)
    except Exception as e:
        print(e)
        return 1
    return 0

def unittest():
    db = load_database("Test")
    db.update("Test", {"Person 1": 2, "Person 2": 4})
    save_database(db)
    db = load_database("Test")
    db.display_entries()
    db.search("Person 1")
    db.search(["Person 1", "Person 2"])


if __name__ == "__main__":
    db = load_database("test3")
    db.display_entries()
    db.search("Ryan Gosling")
    db.search(["Ryan Gosling", "Emma Stone"])
