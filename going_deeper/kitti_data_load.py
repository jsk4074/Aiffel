# takes really long....
import tensorflow_datasets as tfds
import os
import urllib3


urllib3.disable_warnings()

data_dir = os.path.join(os.getenv("HOME"), "desktop/workspace/aiffel_git/data/GD8")

(ds_train, ds_test), ds_info = tfds.load(
    'kitti',
    data_dir=data_dir,
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True,
)

print('Data loaded')