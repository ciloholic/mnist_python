from PIL import Image
from tqdm import tqdm
import os
import gzip
import numpy as np

key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

num = {
    'train': 60000,
    'test': 10000
}
img_dim = (28, 28)
img_size = 784


def _load_label(file_name):
    '''ラベル解凍&読み込み'''
    with gzip.open(file_name, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels


def _load_img(file_name):
    '''画像解凍&読み込み'''
    with gzip.open(file_name, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, img_size)


def _convert_numpy():
    '''ラベル&画像読み込み'''
    return {
        'train_img': _load_img(key_file['train_img']),
        'train_label': _load_label(key_file['train_label']),
        'test_img': _load_img(key_file['test_img']),
        'test_label': _load_label(key_file['test_label'])
    }


def run():
    '''トレーニング&テスト情報分割'''
    dataset = _convert_numpy()
    for case in ['train', 'test']:
        pbar = tqdm(total=num[case])
        for i, img in enumerate(dataset[f'{case}_img']):
            label = dataset[f'{case}_label'][i]
            directory = f'{case}_img/{label}'
            if not os.path.exists(directory):
                os.makedirs(directory)
            pil = Image.fromarray(np.uint8(img.reshape(img_dim)))
            # (train|test)_img/[0-9]/[0-9]{5}.png
            pil.save(f'{case}_img/{label}/{i:05}.png')
            pbar.update(1)
        pbar.close()


if __name__ == '__main__':
    run()
