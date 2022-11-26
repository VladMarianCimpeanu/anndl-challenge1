import splitfolders
import os
#import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
import random
import time


METADATA = os.path.dirname(os.path.abspath(__file__)) + "/../metadata_validation.txt"

def split_folders(split_val: float, log=True, override_metadata=False, seed=100):
    """
    This method is used to split the dataset folder in train folder and validation folder.
    """
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
    splitfolders.ratio(data_path, output=f"{main_path}/../", ratio=(split_val, 1 - split_val, 0), seed=seed)
    with open(METADATA, "w") as writer:
        writer.write(f"current validation split: {split_val}")


def create_1_vs_all_ds():
    """
    This method creates two folders: one containing the training folder, the other the validation folder.
    Both of them contains 2 folders: the first containing Species1 images, the second one images belonging to 
    othe species. This method is used to creating datasets for training 1_vs_ALL classification networks.
    """
    # pointing to the right folder
    relative_path = os.path.dirname(os.path.abspath(__file__))
    train_path = f"{relative_path}\\..\\train_oversampled"
    val_path = f"{relative_path}\\..\\val"
    labels = os.listdir(train_path)
    labels.sort()

    # creating new training folder with oversampling
    os.mkdir(f"{relative_path}\\..\\train_1_vs_all")
    os.mkdir(f"{relative_path}\\..\\val_1_vs_all")

    #creating training folder for species1
    print("creating training folder for species1...")
    os.mkdir(f"{relative_path}\\..\\train_1_vs_all\\Species1")
    from_directory = f"{relative_path}\\..\\train_oversampled\\Species1"
    to_directory = f"{relative_path}\\..\\train_1_vs_all\\Species1"
    files = os.listdir(from_directory)
    for img in files:
        shutil.copy(f"{from_directory}\\{img}", to_directory)    

    # creating validation folder for species1
    print("creating validation folder for species1...")
    os.mkdir(f"{relative_path}\\..\\val_1_vs_all\\Species1")
    from_directory = f"{relative_path}\\..\\val\\Species1"
    to_directory = f"{relative_path}\\..\\val_1_vs_all\\Species1"
    files = os.listdir(from_directory)
    for img in files:
        shutil.copy(f"{from_directory}\\{img}", to_directory) 

    # creating folders for other species
    os.mkdir(f"{relative_path}\\..\\train_1_vs_all\\Other")
    os.mkdir(f"{relative_path}\\..\\val_1_vs_all\\Other")
    
    # creating samples of category "Other" for training
    print("creating training folder for Other category...") 
    for label in labels[1:]:
        # copy subdirectory
        print(f"importing {label} for training...") 
        from_directory = f"{train_path}\\{label}"
        to_directory = f"{relative_path}\\..\\train_1_vs_all\\Other\\"
        files = os.listdir(from_directory)
        for img in files:
            shutil.copy(f"{from_directory}\\{img}", to_directory)
            os.rename(f"{to_directory}\\{img}", f"{to_directory}\\{label}_{img}")
        
    # creating samples of category "Other" for validation
    print("creating validation folder for Other category...")      
    for label in labels[1:]:
        print(f"importing {label} for validation...")
        # copy subdirectory 
        from_directory = f"{val_path}\\{label}"
        to_directory = f"{relative_path}\\..\\val_1_vs_all\\Other\\"
        files = os.listdir(from_directory)
        for img in files:
            shutil.copy(f"{from_directory}\\{img}", to_directory)
            os.rename(f"{to_directory}\\{img}", f"{to_directory}\\{label}_{img}")


def over_sampling():
    """
    This method is used to split the dataset folder in train folder and validation folder.
    Species with frequencies less than 0.8 * max_frequency are oversampled to reach max frequency.
    max_frequency is the number of images belonging to the most frequent dataset.
    """
    # pointing to the right folder
    relative_path = os.path.dirname(os.path.abspath(__file__))
    train_path = f"{relative_path}\\..\\train"
    labels = os.listdir(train_path)

    counts = {}
    # computing frequency for each class
    for label in labels:
        counts[label] = len(os.listdir(f"{train_path}/{label}"))
        print(f"visited {label}")
    highest_frequency = max(counts.values())

    # creating new training folder with oversampling
    os.mkdir(f"{relative_path}\\..\\train_oversampled")
    for label in labels:
        os.mkdir(f"{relative_path}\\..\\train_oversampled\\{label}")
        # copy subdirectory 
        from_directory = f"{relative_path}\\..\\train\\{label}"
        to_directory = f"{relative_path}\\..\\train_oversampled\\{label}"
        files = os.listdir(from_directory)
        for img in files:
            shutil.copy(f"{from_directory}\\{img}", to_directory)
        if counts[label] < 0.8 * highest_frequency: # acceptable threashold
            for _ in range(highest_frequency - counts[label]):
                sample = random.choice(files)
                shutil.copyfile(f"{to_directory}\\{sample}", f"{to_directory}\\{random.randint(0, 10000)}{sample}")
            print(label, len(os.listdir(to_directory)), counts[label], highest_frequency)
            

if __name__ == "__main__":
    split_folders(0.8)
    over_sampling()