import pandas as pd

df = pd.read_csv("result-rj.csv")

df.head()

"""## Separando similar e não similar
Usando distância >= a 0.25 como parâmetro
"""

similar_images_df = df[df['distance'] <= 0.25]

unique_similar_original_photos = set(similar_images_df['original_photo'])
all_unique_original_photos = set(df['original_photo'])

non_similar_photos = all_unique_original_photos - unique_similar_original_photos
non_similar_counts = df[df['original_photo'].isin(non_similar_photos)].groupby('original')['original_photo'].nunique()

non_similar_counts.sort_values(ascending=False)

"""## Percentual de "Novidades" idioma
Imagens únicas

"""

total_counts = df.groupby('original')['original_photo'].nunique()

percentage_non_similar = (non_similar_counts / total_counts) * 100

percentage_non_similar_sorted = percentage_non_similar.sort_values(ascending=False)
percentage_non_similar_sorted

"""## Artigos mais similares entre si
Comparação em números absolutos

"""

similar_images_grouped = similar_images_df.groupby(['original', 'compare']).size().reset_index(name='count')

sorted_similar_images = similar_images_grouped.sort_values(by='count', ascending=False)

sorted_similar_images

"""## Artigos similares em percentual"""

average_image_counts = []
for idx, row in similar_images_grouped.iterrows():
    avg_count = (total_counts[row['original']] + total_counts[row['compare']]) / 2
    average_image_counts.append(avg_count)

percentage_similar_images = (similar_images_grouped['count'] / average_image_counts) * 100

similar_images_grouped['percentage'] = percentage_similar_images
sorted_percentage_similar = similar_images_grouped.sort_values(by='percentage', ascending=False)

sorted_percentage_similar

"""## Imagens que mais aparecem"""

image_languages = similar_images_df.groupby('original_photo')['compare'].unique().reset_index(name='languages')

image_languages['num_languages'] = image_languages['languages'].apply(len)

sorted_image_languages = image_languages.sort_values(by='num_languages', ascending=False)

top_images = sorted_image_languages.head()
top_images

"""## Grafo de similaridade"""

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

for language in total_counts.index:
    G.add_node(language)

for idx, row in sorted_percentage_similar.iterrows():
    G.add_edge(row['original'], row['compare'], weight=row['percentage'])

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)
edges = G.edges(data=True)

nx.draw_networkx_nodes(G, pos, node_size=700)
nx.draw_networkx_edges(G, pos, edgelist=edges, width=[d['weight'] / 15 for _, _, d in edges])
nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

plt.title("Grafo de linguagens similares")
plt.show()

"""## Matriz de Adjacência"""

import seaborn as sns

adj_matrix = similar_images_df.pivot_table(index='original', columns='compare', values='distance', aggfunc='count').fillna(0)
adj_matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(adj_matrix, annot=True, cmap='YlGnBu', cbar=True, linewidths=.5)
# plt.title('Heatmap de similaridade entre linguagens')
# plt.show()

percentage_similar = []

for original in similar_images_df['original'].unique():
    for compare in similar_images_df['compare'].unique():
        if original != compare:
            total_original = df[df['original'] == original].shape[0]
            total_compare = df[df['compare'] == compare].shape[0]
            similar_count = similar_images_df[(similar_images_df['original'] == original) & (similar_images_df['compare'] == compare)].shape[0]
            percentage = (similar_count / ((total_original + total_compare) / 2)) * 100
            percentage_similar.append({'original': original, 'compare': compare, 'percentage': percentage})

percentage_similar_df = pd.DataFrame(percentage_similar)

adj_matrix_percentage = percentage_similar_df.pivot(index='original', columns='compare', values='percentage').fillna(0)

plt.figure(figsize=(10, 8))
sns.heatmap(adj_matrix_percentage, annot=True, cmap='YlGnBu', cbar=True, linewidths=.5, fmt=".2f")
plt.title('Heatmap Similaridade')
plt.show()

"""# Sistema de pontuação

### Pontuação = (novidades x 2) + (total - novidades)
"""

weight_unique = 2
weight_repeated = 1

languages_score = {}
for language in total_counts.index:
    total_content = total_counts[language]
    unique_content = non_similar_counts.get(language, 0)
    repeated_content = total_content - unique_content
    score = (unique_content * weight_unique) + (repeated_content * weight_repeated)
    languages_score[language] = score

sorted_languages_score = dict(sorted(languages_score.items(), key=lambda item: item[1], reverse=True))
sorted_languages_score

"""### Decaimento Exponencial para Conteúdo Repetido
uma imagem que aparece em 7 idiomas teria menos peso do que uma que aparece em apenas 2 idiomas
"""

import numpy as np

decay_scores = {}
for language in total_counts.index:
    total_content = total_counts[language]
    unique_content = non_similar_counts.get(language, 0)
    repeated_content = total_content - unique_content
    decay_value = np.exp(-repeated_content/10)
    score = unique_content + decay_value * repeated_content
    decay_scores[language] = score

sorted_decay_scores = dict(sorted(decay_scores.items(), key=lambda item: item[1], reverse=True))
sorted_decay_scores

"""### Diversidade de Idiomas
 pontuação extra para um idioma que tem conteúdo similar com uma diversidade maior de outros idiomas. Isso pode indicar um equilíbrio de conteúdo entre diferentes culturas e perspectivas.
"""

diversity_scores = {}
for language in total_counts.index:
    diversity = len(df[df['original'] == language]['compare'].unique())
    score = total_counts[language] * diversity
    diversity_scores[language] = score

sorted_diversity_scores = dict(sorted(diversity_scores.items(), key=lambda item: item[1], reverse=True))
sorted_diversity_scores