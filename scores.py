import numpy as np
import pandas as pd
import math
from hashlib import sha256

def new_score(df):
    df['id'] = df['original_photo'].apply(lambda x: sha256(x.encode('utf-8')).hexdigest())
    
    filtered_df = df[df['distance'] <= 0.25]

    filtered_no_eq_df = df[~df['id'].isin(filtered_df['id']) & df['distance'] > 0.25]
    filtered_no_eq_df = filtered_no_eq_df.drop_duplicates(subset=['id'])


    uniques_df = pd.DataFrame({
        'original_photo': filtered_no_eq_df['original_photo'],
        'languages':  filtered_no_eq_df['original'].apply(lambda x: [x]),
        'num_languages': 1
    }).reset_index()

    similar_df = (filtered_df.groupby('original_photo')
             .apply(lambda x: pd.Series({
                 'languages': list(set(x['original']).union(set(x['compare']))),
                 'num_languages': len(list(set(x['original']).union(set(x['compare'])))),
                 'id': x['id']
                 }))
             .reset_index())
    
    grouped_df = pd.concat([similar_df, uniques_df])
    sorted_df = grouped_df.sort_values(by='num_languages', ascending=False)

    def parabola_score(x, N_LANGS = 8):
        if x <= (N_LANGS // 2):
            return np.exp(-(x - 1) / math.pi)
        return np.exp((x) / math.pi) / np.exp(N_LANGS / math.pi)


    def decay_score(x):
        return np.exp(-(x - 1) / math.pi)

    def growth_score(x, N_LANGS = 8):
        return np.exp((x) / math.pi) / np.exp(N_LANGS / math.pi)
    
    sorted_df['decay'] = sorted_df['num_languages'].apply(lambda x: decay_score(x))
    sorted_df['growth'] = sorted_df['num_languages'].apply(lambda x: growth_score(x))
    sorted_df['parabola'] = sorted_df['num_languages'].apply(lambda x: parabola_score(x))

    langs_to_check = ['pt', 'en', 'es', 'de', 'it', 'ru', 'zh', 'fr']

    dict_scores_decay = {}
    dict_scores_growth = {}
    dict_scores_parabola = {}


    for lang in langs_to_check:
        lang_df = sorted_df[sorted_df['languages'].apply(lambda x: lang in x)]

        dict_scores_decay[lang] = lang_df['decay'].sum()
        dict_scores_growth[lang] = lang_df['growth'].sum()
        dict_scores_parabola[lang] = lang_df['parabola'].sum()

    scores = {
        'decay': dict_scores_decay,
        'growth': dict_scores_growth,
        'parabola': dict_scores_parabola
    }

    return scores

def simple_score(df):
    similar_images_df = df[df['distance'] <= 0.25]

    unique_similar_original_photos = set(similar_images_df['original_photo'])
    all_unique_original_photos = set(df['original_photo'])

    non_similar_photos = all_unique_original_photos - unique_similar_original_photos

    total_counts = df.groupby('original')['original_photo'].nunique()
    non_similar_counts = df[df['original_photo'].isin(non_similar_photos)].groupby(
        'original')['original_photo'].nunique()

    weight_unique = 2
    weight_repeated = 1

    languages_score = {}
    for language in total_counts.index:
        total_content = total_counts[language]
        unique_content = non_similar_counts.get(language, 0)
        repeated_content = total_content - unique_content
        score = (unique_content * weight_unique) + \
            (repeated_content * weight_repeated)
        languages_score[language] = score

    sorted_languages_score = dict(
        sorted(languages_score.items(), key=lambda item: item[1], reverse=True))
    return sorted_languages_score


def decay_score(df):
    similar_images_df = df[df['distance'] <= 0.25]

    unique_similar_original_photos = set(similar_images_df['original_photo'])
    all_unique_original_photos = set(df['original_photo'])

    non_similar_photos = all_unique_original_photos - unique_similar_original_photos

    total_counts = df.groupby('original')['original_photo'].nunique()
    non_similar_counts = df[df['original_photo'].isin(non_similar_photos)].groupby(
        'original')['original_photo'].nunique()

    decay_scores = {}
    for language in total_counts.index:
        total_content = total_counts[language]
        unique_content = non_similar_counts.get(language, 0)
        repeated_content = total_content - unique_content
        decay_value = np.exp(-repeated_content/10)
        score = unique_content + decay_value * repeated_content
        decay_scores[language] = score

    sorted_decay_scores = dict(
        sorted(decay_scores.items(), key=lambda item: item[1], reverse=True))
    return sorted_decay_scores


def diversity_score(df):
    total_counts = df.groupby('original')['original_photo'].nunique()

    diversity_scores = {}
    for language in total_counts.index:
        diversity = len(df[df['original'] == language]['compare'].unique())
        score = total_counts[language] * diversity
        diversity_scores[language] = score

    sorted_diversity_scores = dict(
        sorted(diversity_scores.items(), key=lambda item: item[1], reverse=True))
    return sorted_diversity_scores




def parabola_score(df):
    total_counts = df.groupby('original')['original_photo'].nunique()
    N = len(total_counts.index)

    parabolic_scores = {}
    for language in total_counts.index:
        content_count = total_counts[language]
        score = (content_count - 1) * (content_count - N)
        parabolic_scores[language] = score

    sorted_decay_scores = dict(
        sorted(parabolic_scores.items(), key=lambda item: item[1], reverse=True))
    return sorted_decay_scores



