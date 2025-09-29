import os
import shutil
import random

def split_dataset(main_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    os.chdir(main_dir)
    

    labels = [label for label in os.listdir() if os.path.isdir(label)]
    
    if os.path.isdir('train') is False:
        os.mkdir('train')
        os.mkdir('valid')
        os.mkdir('test')

        for label in labels:

            if not os.path.isdir(f'train/{label}'):
                shutil.move(label, 'train')

            os.mkdir(f'valid/{label}')
            os.mkdir(f'test/{label}')

            label_path = f'train/{label}'
            images = os.listdir(label_path)
            total_images = len(images)

            train_count = int(total_images * train_ratio)
            valid_count = int(total_images * valid_ratio)

            random.shuffle(images)
            train_images = images[:train_count]
            valid_images = images[train_count:train_count + valid_count]
            test_images = images[train_count + valid_count:]

            for image in valid_images:
                shutil.move(f'{label_path}/{image}', f'valid/{label}')

            for image in test_images:
                shutil.move(f'{label_path}/{image}', f'test/{label}')

    os.chdir('../..')


main_directory = 'dying_leaves'
split_dataset(main_directory)
