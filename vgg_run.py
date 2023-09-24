from keras.applications.vgg19 import VGG19, preprocess_input
import keras.utils as image
from scipy.spatial import distance
import numpy as np
import os
import itertools
import concurrent.futures

model = VGG19(weights='imagenet', include_top=False)

def get_img_files(base_path):
    file_list = []

    folder_path = f'data/paraiba/{base_path}'


    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_list.append(file_name)
    return (base_path, sorted(file_list))

def extract_features_vgg19(img_path):
    img = image.load_img(f'data/paraiba/{img_path}', target_size=(224, 224))

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
        list_to_compare.append(((lang, compare_lang), (photo, compare_photo)))


def compare_images(value_to_compare):
    langs, imgs = value_to_compare
    
    img_original, img_compare = imgs

    res = compare_two_images_vgg19(f'{langs[0]}/{img_original}', f'{langs[1]}/{img_compare}')

    print((langs, imgs, res))

    return (langs, imgs, res)


with concurrent.futures.ProcessPoolExecutor() as executor:
    resultados = list(executor.map(compare_images, list_to_compare))

    print(resultados)
# compare_list = []


# for item in img_lang_files:
#   current_lang = item[0]
#   current_items = item[1]


#   for subitem in img_lang_files:
#     if subitem[0] != current_lang:
#       c_product = list(itertools.product(current_items, subitem[1]))
#       t_compare = (current_lang, subitem[0], c_product)

#       compare_list.append(t_compare)
  
# for compare in compare_list:
#   for img in compare[2]:
#     sim = {
#       f'{compare[0]}': '',
#       f'{compare[1]}': '',
#     }

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#       try:

#         future = executor.submit(compare_two_images_vgg19, img[0], img[1])
#         s = future.result()

#         if s < 0.5:
#           if not values


#       values = ()

#       for future in concurrent.futures.as_completed(future_to_img):
#           pt_img = future_to_img[future]
#           try:
#               s, pt_img = future.result()
#               if s < 0.5:
#                 if not values:
#                   values = (f'pt-pele/{pt_img}', s)
#                 else:
#                   if s < values[1]:
#                     values = values = (f'pt-pele/{pt_img}', s)
#           except Exception as exc:
#               print('%r generated an exception: %s' % (pt_img, exc))

#       if values:
#         sim['pt'] = values[0]
#         sim['s'] = values[1]

