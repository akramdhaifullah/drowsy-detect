import os

from sklearn.model_selection import train_test_split


def subject_wise_split(image_paths, test_size=0.2, random_state=42):

    subjects = [os.path.basename(os.path.dirname(path)) for path in image_paths]

    unique_subjects = sorted(set(subjects))

    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=test_size, random_state=random_state
    )

    train_files = []
    test_files = []

    for path in image_paths:
        subject = os.path.basename(os.path.dirname(path))

        if subject in train_subjects:
            train_files.append(path)
        else:
            test_files.append(path)

    return train_files, test_files
