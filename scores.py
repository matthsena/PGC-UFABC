import numpy as np


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
