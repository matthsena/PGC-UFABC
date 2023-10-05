from keras.applications.vgg19 import VGG19, preprocess_input
import keras.utils as image
from scipy.spatial import distance
import numpy as np
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import hashlib

model = VGG19(weights='imagenet', include_top=False)

base_folder = 'data/rio/'

def get_img_files(base_path):
    file_list = []

    folder_path = f'{base_folder}{base_path}'


    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_list.append(file_name)
    return (base_path, sorted(file_list))

def extract_features_vgg19(img_path):
    img = image.load_img(f'{base_folder}{img_path}', target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    features = model.predict(x)
    features = features.flatten()

    return features

def compare_two_images_vgg19(img1_path, img2_path):
  try:
    print(f'compare {img1_path} with {img2_path}')
    features1 = extract_features_vgg19(img1_path)
    features2 = extract_features_vgg19(img2_path)

    return distance.cosine(features1, features2)
  except Exception as e:
    print(f'ERROR: {img1_path} -> {img2_path} ==> {e}')


langs_to_check = ['pt', 'en', 'es', 'de', 'it', 'ru', 'zh', 'fr']

 
img_lang_files = list(map(get_img_files, langs_to_check))

list_to_compare = []

for item in img_lang_files:
  lang, photos = item

  compare_with = [img_folder for img_folder in img_lang_files if img_folder[0] != lang]

  for photo in photos:
    for compare in compare_with:
      compare_lang, compare_photos = compare

      for compare_photo in compare_photos:
        str_representation = ''.join(sorted([lang, compare_lang, photo, compare_photo]))
        
        hash_object = hashlib.sha256(str_representation.encode())
        hash_hex = hash_object.hexdigest()

        list_to_compare.append((hash_hex, (lang, compare_lang), (photo, compare_photo)))


result_list = []


seen = set()
filtered_list = []

for item in list_to_compare:
    if item[0] not in seen:
        seen.add(item[0])
        filtered_list.append((item[1], item[2]))


with ThreadPoolExecutor() as executor:
    futures = []
    for item in filtered_list:
        lang, img = item
        future = executor.submit(compare_two_images_vgg19, f'{lang[0]}/{img[0]}', f'{lang[1]}/{img[1]}')
        futures.append(future)

    for future, item in zip(futures, filtered_list):
        lang, img = item
        
        result_list.append({
           'original': lang[0],
           'compare': lang[1],
           'original_photo': img[0],
           'compare_photo': img[1],
           'distance': future.result()	
        })

        print(f'{lang[0]}/{img[0]} -> {lang[1]}/{img[1]}: {future.result()}')



pd.DataFrame(result_list).to_csv('result-rj.csv', index=False)