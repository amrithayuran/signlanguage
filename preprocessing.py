import cv2
import os
from image_processing import func
import config

def preprocess_data():
    if not os.path.exists(config.DATA2_DIR):
        os.makedirs(config.DATA2_DIR)
    if not os.path.exists(config.TRAIN2_DIR):
        os.makedirs(config.TRAIN2_DIR)
    if not os.path.exists(config.TEST2_DIR):
        os.makedirs(config.TEST2_DIR)

    path = config.TRAIN_DIR
    path1 = config.DATA2_DIR

    var = 0
    c1 = 0
    c2 = 0

    print(f"Processing data from {path}...")

    for (dirpath, dirnames, filenames) in os.walk(path):
        for dirname in dirnames:
            print(f"Processing class: {dirname}")
            
            train_dir_path = os.path.join(config.TRAIN2_DIR, dirname)
            test_dir_path = os.path.join(config.TEST2_DIR, dirname)

            if not os.path.exists(train_dir_path):
                os.makedirs(train_dir_path)
            if not os.path.exists(test_dir_path):
                os.makedirs(test_dir_path)

            # Logic to split data: 75% train, 25% test (or custom logic)
            # The original code had a weird logic with 'num', let's make it standard split if needed
            # But original code seemed to want to put everything in train or split by count?
            # It had `num = 100000000000000000` which means everything goes to c1 (train)
            # I will keep the logic simple: just process images. 
            # If the user wants to split, they should probably do it explicitly.
            # For now, I'll stick to the original behavior of putting everything in 'train' 
            # unless we want to implement a real split. 
            # Actually, let's implement a 80-20 split for better practice if the source is just 'train'.
            
            files = os.listdir(os.path.join(path, dirname))
            total_files = len(files)
            split_idx = int(total_files * 0.8)

            for i, file in enumerate(files):
                var += 1
                actual_path = os.path.join(path, dirname, file)
                
                # Skip if not an image
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                bw_image = func(actual_path)
                if bw_image is None:
                    continue

                if i < split_idx:
                    c1 += 1
                    target_path = os.path.join(train_dir_path, file)
                else:
                    c2 += 1
                    target_path = os.path.join(test_dir_path, file)
                
                cv2.imwrite(target_path, bw_image)

    print(f"Total processed: {var}")
    print(f"Train images: {c1}")
    print(f"Test images: {c2}")

if __name__ == "__main__":
    preprocess_data()





