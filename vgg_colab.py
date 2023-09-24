import os
import pandas as pd

def get_img_files(folder_path):
    file_list = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_list.append(file_name)
    return sorted(file_list)

from keras.applications.vgg19 import VGG19, preprocess_input
import keras.utils as image
from scipy.spatial import distance
import numpy as np

model = VGG19(weights='imagenet', include_top=False)

def extract_features_vgg19(img_path):
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    features = model.predict(x)
    features = features.flatten()

    return features

def compare_two_images_vgg19(img1_path, img2_path):
  try:
    features1 = extract_features_vgg19(img1_path)
    features2 = extract_features_vgg19(img2_path)

    return distance.cosine(features1, features2)
  except Exception as e:
    print(f'ERROR: {img1_path} -> {img2_path} ==> {e}')

img_en = get_img_files('data/paraiba/en')
img_pt = get_img_files('data/paraiba/pt')

comparison  = []

import concurrent.futures

def compare_images(en_img, pt_img):
    s = compare_two_images_vgg19(f'data/paraiba/en/{en_img}', f'data/paraiba/pt/{pt_img}')
    print(f'{en_img} -> {pt_img} == {s}')
    return s, pt_img

comparison = []

for en_img in img_en:
    sim = {
        'en': f'data/paraiba/en/{en_img}'
    }

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_img = {executor.submit(compare_images, en_img, pt_img): pt_img for pt_img in img_pt}

        values = ()

        for future in concurrent.futures.as_completed(future_to_img):
            pt_img = future_to_img[future]
            try:
                s, pt_img = future.result()
                if s < 0.25:
                  if not values:
                    values = (f'data/paraiba/pt/{pt_img}', s)
                  else:
                    if s < values[1]:
                      values = values = (f'data/paraiba/pt/{pt_img}', s)
            except Exception as exc:
                print('%r generated an exception: %s' % (pt_img, exc))

        if values:
          sim['pt'] = values[0]
          sim['s'] = values[1]

    print(sim)
    comparison.append(sim)

for pt_img in img_pt:
    sim = {
        'pt': f'data/paraiba/pt/{pt_img}'
    }

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_img = {executor.submit(compare_images, en_img, pt_img): en_img for en_img in img_en}

        values = ()

        for future in concurrent.futures.as_completed(future_to_img):
            en_img = future_to_img[future]
            try:
                s, en_img = future.result()
                if s < 0.25:
                  if not values:
                    values = (f'data/paraiba/en/{en_img}', s)
                  else:
                    if s < values[1]:
                      values = values = (f'data/paraiba/en/{en_img}', s)
            except Exception as exc:
                print('%r generated an exception: %s' % (en_img, exc))

        if values:
          sim['en'] = values[0]
          sim['s'] = values[1]

    print(sim)
    comparison.append(sim)


df_comparison = pd.DataFrame(comparison)
df_comparison = df_comparison.fillna('')

print(df_comparison)

df_comparison.to_csv('paraiba-pt-en.csv', index=False)
