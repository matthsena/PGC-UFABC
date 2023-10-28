from keras.applications.vgg19 import VGG19, preprocess_input
import keras.utils as image
from scipy.spatial import distance
import numpy as np
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
from scores import simple_score, decay_score, diversity_score
import sys

start_time = time.time()

model = None

base_folder = f'data/{sys.argv[1]}/'


def load_model():
    global model
    if model is None:
        model = VGG19(weights='imagenet', include_top=False)


def get_img_files(base_path):
    folder_path = os.path.join(base_folder, base_path)
    file_list = [file_name for file_name in os.listdir(
        folder_path) if os.path.isfile(os.path.join(folder_path, file_name))]
    return (base_path, sorted(file_list))


def extract_features_vgg19(img_path):
    img = image.load_img(os.path.join(
        base_folder, img_path), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()


def compare_features(features1, features2):
    return distance.cosine(features1, features2)


langs_to_check = ['pt', 'en', 'es', 'de', 'it', 'ru', 'zh', 'fr']

img_lang_files = list(map(get_img_files, langs_to_check))

load_model()

# Extract features for all images first
features_dict = {}
for lang, photos in img_lang_files:
    for photo in photos:
        img_path = os.path.join(lang, photo)
        features_dict[img_path] = extract_features_vgg19(img_path)

list_to_compare = []
seen = set()

for lang, photos in img_lang_files:
    for other_lang, other_photos in img_lang_files:
        if lang != other_lang:
            for photo in photos:
                for other_photo in other_photos:
                    sorted_items = tuple(
                        sorted([lang, other_lang, photo, other_photo]))
                    if sorted_items not in seen:
                        seen.add(sorted_items)
                        list_to_compare.append(
                            ((lang, other_lang), (photo, other_photo)))

result_list = []

with ThreadPoolExecutor() as executor:
    futures = []
    for (lang1, lang2), (img1, img2) in list_to_compare:
        future = executor.submit(compare_features, features_dict[os.path.join(
            lang1, img1)], features_dict[os.path.join(lang2, img2)])
        futures.append((future, (lang1, lang2, img1, img2)))

    for future, (lang1, lang2, img1, img2) in futures:
        result_list.append({
            'article': sys.argv[1],
            'original': lang1,
            'compare': lang2,
            'original_photo': img1,
            'compare_photo': img2,
            'distance': future.result()
        })
        # print(f'{lang1}/{img1} -> {lang2}/{img2}: {future.result()}')

df_result = pd.DataFrame(result_list)

print('Score simples')
print(simple_score(df_result))

print('Score decaimento exponencial')
print(decay_score(df_result))

print('Score de diversidade')
print(diversity_score(df_result))


# pd.DataFrame(result_list).to_csv('result-rj.csv', index=False)

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Total time taken: {elapsed_time:.2f} seconds")
