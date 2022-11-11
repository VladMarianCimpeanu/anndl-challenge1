import splitfolders
import os

METADATA = os.path.dirname(os.path.abspath(__file__)) + "/../metadata_validation.txt"

def split_folders(split_val: float, log=False, override_metadata=False, seed=100):
    main_path = os.path.dirname(os.path.abspath(__file__))
    if "metadata_validation.txt" in os.listdir(f"{main_path}/.."):
        with open(METADATA, "r") as reader:
            content = reader.readline()
            past_validation_split = content.split(" ")[-1]
            if past_validation_split == split_val and not override_metadata:
                if log:
                    print("Dataset already splitted")
                return
    data_path = f"{main_path}/../training_data_final/"
    splitfolders.ratio(data_path, output=f"{main_path}/../", ratio=(1 - split_val, split_val, 0), seed=seed)
    with open(METADATA, "w") as writer:
        writer.write(f"current validation split: {split_val}")


if __name__ == "__main__":
    split_folders(0.8)
