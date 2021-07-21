# [run] python ./3_prepare_data.py

import shutil
import zipfile
import os

zip_file = zipfile.ZipFile('./modelnet40_ply_hdf5_2048.zip')
zip_file.extractall()

os.mkdir('./data')

shutil.move('./modelnet40_ply_hdf5_2048', './data')

try:
    os.remove('./modelnet40_ply_hdf5_2048.zip')
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))