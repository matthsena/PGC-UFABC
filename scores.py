import numpy as np
import pandas as pd
import math
from hashlib import sha256

class ScoreCalculator:
    MAX_LANGS = 8
    LANGUAGES_TO_CHECK = ['pt', 'en', 'es', 'de', 'it', 'ru', 'zh', 'fr']

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.dataframe['id'] = self.dataframe['original_photo'].apply(lambda x: sha256(x.encode('utf-8')).hexdigest())

    def calculate_score(self):
        close_distance_df, far_distance_df = self._split_dataframe_by_distance()
        unique_df, similar_df = self._create_unique_and_similar_df(far_distance_df, close_distance_df)
        sorted_df = self._combine_and_sort_df(unique_df, similar_df)
        scores = self._calculate_scores(sorted_df)
        return scores

    def _split_dataframe_by_distance(self):
        close_distance_df = self.dataframe[self.dataframe['distance'] <= 0.25]
        far_distance_df = self.dataframe[~self.dataframe['id'].isin(close_distance_df['id']) & self.dataframe['distance'] > 0.25]
        far_distance_df = far_distance_df.drop_duplicates(subset=['id'])
        return close_distance_df, far_distance_df

    def _create_unique_and_similar_df(self, far_distance_df, close_distance_df):
        unique_df = pd.DataFrame({
            'original_photo': far_distance_df['original_photo'],
            'languages':  far_distance_df['original'].apply(lambda x: [x]),
            'num_languages': 1
        }).reset_index()

        similar_df = (close_distance_df.groupby('original_photo')
                 .apply(lambda x: pd.Series({
                     'languages': list(set(x['original']).union(set(x['compare']))),
                     'num_languages': len(list(set(x['original']).union(set(x['compare'])))),
                     'id': x['id']
                     }))
                 .reset_index())
        return unique_df, similar_df

    def _combine_and_sort_df(self, unique_df, similar_df):
        combined_df = pd.concat([similar_df, unique_df])
        sorted_df = combined_df.sort_values(by='num_languages', ascending=False)
        return sorted_df

    def _calculate_scores(self, sorted_df):
        sorted_df['decay'] = sorted_df['num_languages'].apply(self._calculate_decay_score)
        sorted_df['growth'] = sorted_df['num_languages'].apply(self._calculate_growth_score)
        sorted_df['parabola'] = sorted_df['num_languages'].apply(self._calculate_parabola_score)
        sorted_df['simple'] = sorted_df['num_languages'].apply(self._calculate_simple_score)

        decay_scores, growth_scores, parabola_scores, simple_scores = self._calculate_language_scores(sorted_df)

        scores = {
            'decay': decay_scores,
            'growth': growth_scores,
            'parabola': parabola_scores,
            'simple': simple_scores,
        }
        return scores

    def _calculate_decay_score(self, num_languages):
        return np.exp(-(num_languages - 1) / math.pi)

    def _calculate_growth_score(self, num_languages):
        return np.exp(num_languages / math.pi) / np.exp(self.MAX_LANGS / math.pi)

    def _calculate_parabola_score(self, num_languages):
        if num_languages <= (self.MAX_LANGS // 2):
            return np.exp(-(num_languages - 1) / math.pi)
        return np.exp(num_languages / math.pi) / np.exp(self.MAX_LANGS / math.pi)

    def _calculate_simple_score(self, num_languages):
        return 2 if num_languages == 1 else 1

    def _calculate_language_scores(self, sorted_df):
        decay_scores = {}
        growth_scores = {}
        parabola_scores = {}
        simple_scores = {}

        for language in self.LANGUAGES_TO_CHECK:
            language_df = sorted_df[sorted_df['languages'].apply(lambda x: language in x)]

            decay_scores[language] = language_df['decay'].sum()
            growth_scores[language] = language_df['growth'].sum()
            parabola_scores[language] = language_df['parabola'].sum()
            simple_scores[language] = language_df['simple'].sum()

        return decay_scores, growth_scores, parabola_scores, simple_scores
