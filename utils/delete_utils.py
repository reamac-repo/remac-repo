import os

def delete_empty_folders_in_current_dir():
    # get current directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # travel through all items in the directory
    for item in os.listdir(script_dir):
        item_path = os.path.join(script_dir, item)

        # check whether it is a directory and whether it is empty
        if os.path.isdir(item_path) and not os.listdir(item_path):
            print(f"delete empty directory: {item_path}")
            os.rmdir(item_path)

if __name__ == "__main__":
    delete_empty_folders_in_current_dir()
    print("empty folder cleanup completed!")