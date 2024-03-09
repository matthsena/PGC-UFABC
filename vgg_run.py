import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
from scipy.spatial import distance
from itertools import combinations, product
from models.cv.vgg19 import FeatureExtractor
from scores import new_score

import concurrent.futures

class ImageFeatureExtractor:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.feature_extractor = FeatureExtractor()
        self.folders = self.get_folders()

    def get_folders(self):
        return [folder for folder in os.listdir(self.data_directory) if os.path.isdir(os.path.join(self.data_directory, folder))]

    def get_image_files(self, base_folder, base_path):
        folder_path = os.path.join(base_folder, base_path)
        file_list = [file_name for file_name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file_name))]
        return (base_path, sorted(file_list))

    def extract_features(self, base_folder, img_lang_files):
        features_dict = {}
        for lang, photos in img_lang_files:
            for photo in photos:
                img_path = os.path.join(lang, photo)
                img = os.path.join(base_folder, img_path)
                features_dict[img_path] = self.feature_extractor.extract_features(img)
        return features_dict

    def get_comparison_list(self, img_lang_files):
        list_to_compare = []
        seen = set()
        for (lang_1, imgs_1), (lang_2, imgs_2) in combinations(img_lang_files, 2):
            for img_1, img_2 in product(imgs_1, imgs_2):
                items = tuple(sorted([lang_1, lang_2, img_1, img_2]))
                if items not in seen:
                    seen.add(items)
                    list_to_compare.append(((lang_1, lang_2), (img_1, img_2)))
        return list_to_compare

    def get_results(self, base_folder, list_to_compare, features_dict):
        result_list = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.feature_extractor.compare_features, features_dict[os.path.join(lang_1, img1)], features_dict[os.path.join(lang_2, img2)]): (lang_1, lang_2, img1, img2) for (lang_1, lang_2), (img1, img2) in list_to_compare}
            for future in concurrent.futures.as_completed(futures):
                lang_1, lang_2, img1, img2 = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (lang_1, exc))
                else:
                    result_list.append({
                        'article': base_folder,
                        'original': lang_1,
                        'compare': lang_2,
                        'original_photo': img1,
                        'compare_photo': img2,
                        'distance': result
                    })
        return result_list

    def run(self):
        start_time = time.time()
        final_df = pd.DataFrame()
        langs_to_check = ['pt', 'en', 'es', 'de', 'it', 'ru', 'zh', 'fr']
        for folder in self.folders:
            base_folder = f'{self.data_directory}/{folder}/'
            img_lang_files = list(map(lambda base_path: self.get_image_files(base_folder, base_path), langs_to_check))
            features_dict = self.extract_features(base_folder, img_lang_files)
            list_to_compare = self.get_comparison_list(img_lang_files)
            result_list = self.get_results(base_folder, list_to_compare, features_dict)
            df_result = pd.DataFrame(result_list)
            score_z = new_score(df_result)
            df = self.create_dataframe(folder, score_z)
            print(df)
            pd.DataFrame(result_list).to_csv(f'results/{folder}.csv', index=False)
            final_df = pd.concat([final_df, df])
        end_time = time.time()
        elapsed_time = (end_time - start_time) // 60
        print(f"Total time taken: {elapsed_time} minutes and {(end_time - start_time) % 60:.2f} seconds")
        final_df.to_csv('score.csv', index=False)

    def create_dataframe(self, folder, score_z):
        score_types = ['decay', 'growth', 'parabola', 'simple']
        df = pd.DataFrame({f'{folder} Score {score_type}': score_z[score_type] for score_type in score_types}).T
        return df

if __name__ == "__main__":
    data_dir = 'data-quentes'
    extractor = ImageFeatureExtractor(data_dir)
    extractor.run()
