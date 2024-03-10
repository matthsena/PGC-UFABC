import numpy as np
import pandas as pd
from hashlib import sha256


class ScoreCalculator:
    MAX_LANGS = 8
    LANGUAGES_TO_CHECK = ['pt', 'en', 'es', 'de', 'it', 'ru', 'zh', 'fr']

    def __init__(self, dataframe, article_name):
        self.article_name = article_name
        self.dataframe = dataframe
        self.dataframe['id'] = self.dataframe['original_photo'].apply(
            lambda x: sha256(x.encode('utf-8')).hexdigest())

    def calculate_score(self):
        close_distance_df, far_distance_df = self._split_dataframe_by_distance()
        sorted_df = self._create_and_sort_df(
            far_distance_df, close_distance_df)
        return self._calculate_scores(sorted_df)

    def _split_dataframe_by_distance(self):
        close_distance_df = self.dataframe[self.dataframe['distance'] <= 0.25]
        far_distance_df = self.dataframe[~self.dataframe['id'].isin(
            close_distance_df['id']) & self.dataframe['distance'] > 0.25]
        return close_distance_df.drop_duplicates(subset=['id']), far_distance_df

    def _create_and_sort_df(self, far_distance_df, close_distance_df):
        unique_df = far_distance_df.assign(
            languages=far_distance_df['original'].apply(lambda x: [x]), num_languages=1)
        similar_df = close_distance_df.groupby('original_photo').apply(lambda x: pd.Series({
            'languages': list(set(x['original']).union(set(x['compare']))),
            'num_languages': len(list(set(x['original']).union(set(x['compare'])))),
            'id': x['id'].values[0]
        })).reset_index()
        return pd.concat([similar_df, unique_df]).sort_values(by='num_languages', ascending=False)

    def _calculate_scores(self, sorted_df):
        sorted_df = sorted_df.assign(
            decay=sorted_df['num_languages'].apply(
                lambda x: np.exp(-(x - 1) / np.pi)),
            growth=sorted_df['num_languages'].apply(
                lambda x: np.exp(x / np.pi) / np.exp(self.MAX_LANGS / np.pi)),
            parabola=sorted_df['num_languages'].apply(
                self._calculate_parabola_score),
            simple=sorted_df['num_languages'].apply(
                lambda x: 2 if x == 1 else 1)
        )
        return [{**{lang: sorted_df[sorted_df['languages'].apply(lambda x: lang in x)][score].sum() for lang in self.LANGUAGES_TO_CHECK}, 'type': score, 'article': self.article_name} for score in ['decay', 'growth', 'parabola', 'simple']]

    def _calculate_parabola_score(self, num_languages):
        return np.exp(-(num_languages - 1) / np.pi) if num_languages <= (self.MAX_LANGS // 2) else np.exp(num_languages / np.pi) / np.exp(self.MAX_LANGS / np.pi)
