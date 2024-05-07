import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.decomposition import PCA


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

easy_baseline_df = pd.read_csv("data/easy_poems_baseline.csv")
medium_baseline_df = pd.read_csv("data/medium_poems_baseline.csv")
hard_baseline_df = pd.read_csv("data/hard_poems_baseline.csv")

formal_embeddings = []
informal_embeddings = []
traditional_embeddings = []
modern_embeddings = []
serious_embeddings = []
funny_embeddings = []
romantic_embeddings = []
cynical_embeddings = []
rhythmic_embeddings = []
free_embeddings = []
intense_embeddings = []
relaxed_embeddings = []
emotional_embeddings = []
rational_embeddings = []
profound_embeddings = []
superficial_embeddings = []
expressive_embeddings = []
restrained_embeddings = []
happy_embeddings = []
sad_embeddings = []

def process_baseline_df(df):
    print("Processing baseline df")
    for _, row in df.iterrows():
        embedding = model.encode(row['poem'])
        if row['adjective'] == 'formal':
            formal_embeddings.append(embedding)
        elif row['adjective'] == 'informal':
            informal_embeddings.append(embedding)
        elif row['adjective'] == 'traditional':
            traditional_embeddings.append(embedding)
        elif row['adjective'] == 'modern':
            modern_embeddings.append(embedding)
        elif row['adjective'] == 'serious':
            serious_embeddings.append(embedding)
        elif row['adjective'] == 'funny':
            funny_embeddings.append(embedding)
        elif row['adjective'] == 'romantic':
            romantic_embeddings.append(embedding)
        elif row['adjective'] == 'cynical':
            cynical_embeddings.append(embedding)
        elif row['adjective'] == 'rhythmic':
            rhythmic_embeddings.append(embedding)
        elif row['adjective'] == 'free':
            free_embeddings.append(embedding)
        elif row['adjective'] == 'intense':
            intense_embeddings.append(embedding)
        elif row['adjective'] == 'relaxed':
            relaxed_embeddings.append(embedding)
        elif row['adjective'] == 'emotional':
            emotional_embeddings.append(embedding)
        elif row['adjective'] == 'rational':
            rational_embeddings.append(embedding)
        elif row['adjective'] == 'profound':
            profound_embeddings.append(embedding)
        elif row['adjective'] == 'superficial':
            superficial_embeddings.append(embedding)
        elif row['adjective'] == 'expressive':
            expressive_embeddings.append(embedding)
        elif row['adjective'] == 'restrained':
            restrained_embeddings.append(embedding)
        elif row['adjective'] == 'happy':
            happy_embeddings.append(embedding)
        elif row['adjective'] == 'sad':
            sad_embeddings.append(embedding)

process_baseline_df(easy_baseline_df)
process_baseline_df(medium_baseline_df)
process_baseline_df(hard_baseline_df)

embeddings_list = [
    formal_embeddings,
    informal_embeddings,
    traditional_embeddings,
    modern_embeddings,
    serious_embeddings,
    funny_embeddings,
    romantic_embeddings,
    cynical_embeddings,
    rhythmic_embeddings,
    free_embeddings,
    intense_embeddings,
    relaxed_embeddings,
    emotional_embeddings,
    rational_embeddings,
    profound_embeddings,
    superficial_embeddings,
    expressive_embeddings,
    restrained_embeddings,
    happy_embeddings,
    sad_embeddings
]

embeddings_np_list = [np.array(embeddings) for embeddings in embeddings_list]

np.save("data/embeddings_np_list.npy", embeddings_np_list)

embeddings_np_list = np.load("data/embeddings_np_list.npy")

pca = PCA(n_components=5)

pooled_features = []
for embeddings_np in embeddings_np_list:
    pooled_features.append(pca.fit_transform(embeddings_np).mean(axis=0))

pooled_differences = []
for i in range(0, len(pooled_features), 2):
    difference = pooled_features[i] - pooled_features[i + 1]
    pooled_differences.append(difference)

def compute_pca_similarity(input):
    embedding = model.encode(input)
    new_embedding_np = np.array(embedding)

    new_embedding_pca = pca.transform(new_embedding_np.reshape(1, -1))

    similarities = []
    for difference_vector in pooled_differences:
        similarity = np.dot(new_embedding_pca, difference_vector) / (np.linalg.norm(new_embedding_pca) * np.linalg.norm(difference_vector))
        similarities.append(similarity)

    return similarities

features = [
    'formal',
    'traditional',
    'serious',
    'romantic',
    'rhythmic',
    'intense',
    'emotional',
    'profound',
    'expressive',
    'happy'
]

similarities = compute_pca_similarity('Sunrise reveals, yet falsehood in its prime,\
With gilded charlatan ray, the eternal mime.\
Alas, dawn breaks in treacherous sublime. \
\
Gold vestures draped upon the worlds grime,\
Damned daybreak, hailed as divine.\
Sunrise reveals, yet falsehood in its prime.')
for feature, score in zip(features, similarities):
    print(f"Feature: {feature}, Score: {score}")

similarities = compute_pca_similarity('Mornings speak of you, each dawn anew,\
Courting shadows away, with golden hue.\
My heart awakes with the sunrise view,\
Feeling loves warmth, but its all from you,\
In you, the sunrise, my heart finds its view')
for feature, score in zip(features, similarities):
    print(f"Feature: {feature}, Score: {score}")