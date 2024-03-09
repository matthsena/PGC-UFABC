from keras.applications.vgg19 import VGG19, preprocess_input
from scipy.spatial import distance
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
from scores import new_score
from itertools import combinations, product
from models.cv.vgg19 import FeatureExtractor

vgg19 = FeatureExtractor()


start_time = time.time()

# read all folders names in data folder
data_dir = 'data-quentes'

folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

final_df = pd.DataFrame()

vgg19 = FeatureExtractor()

for folder in folders:
    base_folder = f'data-quentes/{folder}/'


    def get_img_files(base_path):
        folder_path = os.path.join(base_folder, base_path)
        file_list = [file_name for file_name in os.listdir(
            folder_path) if os.path.isfile(os.path.join(folder_path, file_name))]
        return (base_path, sorted(file_list))


    def compare_features(features1, features2):
        return distance.cosine(features1, features2)


    langs_to_check = ['pt', 'en', 'es', 'de', 'it', 'ru', 'zh', 'fr']

    img_lang_files = list(map(get_img_files, langs_to_check))

    # Extract features for all images first
    features_dict = {}
    for lang, photos in img_lang_files:
        for photo in photos:
            img_path = os.path.join(lang, photo)
            img = os.path.join(base_folder, img_path)
            features_dict[img_path] = vgg19.extract_features(img)

    list_to_compare = []
    seen = set()

    for (lang_1, imgs_1), (lang_2, imgs_2) in combinations(img_lang_files, 2):
        for img_1, img_2 in product(imgs_1, imgs_2):
            items = tuple(sorted([lang_1, lang_2, img_1, img_2]))
            if items not in seen:
                seen.add(items)
                list_to_compare.append(((lang_1, lang_2), (img_1, img_2)))

    result_list = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for (lang_1, lang_2), (img1, img2) in list_to_compare:
            future = executor.submit(compare_features, features_dict[os.path.join(
                lang_1, img1)], features_dict[os.path.join(lang_2, img2)])
            futures.append((future, (lang_1, lang_2, img1, img2)))

        for future, (lang_1, lang_2, img1, img2) in futures:
            result_list.append({
                'article': folder,
                'original': lang_1,
                'compare': lang_2,
                'original_photo': img1,
                'compare_photo': img2,
                'distance': future.result()
            })

    df_result = pd.DataFrame(result_list)



    score_z =  new_score(df_result)


    score_decaimento = score_z['decay']
    score_decaimento['article'] = folder
    score_decaimento['type'] = 'decaimento'

    score_growth = score_z['growth']
    score_growth['article'] = folder
    score_growth['type'] = 'growth'

    score_parabola = score_z['parabola']
    score_parabola['article'] = folder
    score_parabola['type'] = 'parabola'

    score_simples = score_z['simple']
    score_simples['article'] = folder
    score_simples['type'] = 'simple'

    df = pd.DataFrame({
        f'{folder} Score simples': score_simples,
        f'{folder} Score decaimento exponencial': score_decaimento,
        f'{folder} Score de growth': score_growth,
        f'{folder} Score de parabola': score_parabola
    }).T  # O .T é para transpor o DataFrame, transformando as colunas em linhas e vice-versa

    print(df)

    pd.DataFrame(result_list).to_csv(f'results/{folder}.csv', index=False)
    # pd.DataFrame(df).to_csv(f'results/score/{folder}.csv', index=False)
    final_df = pd.concat([final_df, df])

end_time = time.time()
elapsed_time = (end_time - start_time) // 60
print(f"Total time taken: {elapsed_time} minutes and {(end_time - start_time) % 60:.2f} seconds")

final_df.to_csv('score.csv', index=False)