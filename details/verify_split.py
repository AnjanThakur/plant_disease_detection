import os
def verify_split(train_dir, test_dir):
    """
    Verifies the train and test datasets.
    """
    train_classes = set(os.listdir(train_dir))
    test_classes = set(os.listdir(test_dir))

    if train_classes != test_classes:
        print("Mismatch in classes between train and test!")
        print(f"Train classes: {train_classes}")
        print(f"Test classes: {test_classes}")
        return

    for class_name in train_classes:
        train_files = set(os.listdir(os.path.join(train_dir, class_name)))
        test_files = set(os.listdir(os.path.join(test_dir, class_name)))

        overlap = train_files & test_files
        if overlap:
            print(f"Duplicates found in class '{class_name}': {overlap}")
        else:
            print(f"Class '{class_name}' is correctly split with no duplicates.")

train_dir = "data/train"
test_dir = "data/test"

verify_split(train_dir, test_dir)
