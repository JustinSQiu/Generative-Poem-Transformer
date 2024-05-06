import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def generate_and_save_embeddings():
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

    formal_embeddings_tensor = torch.tensor(formal_embeddings)
    informal_embeddings_tensor = torch.tensor(informal_embeddings)
    traditional_embeddings_tensor = torch.tensor(traditional_embeddings)
    modern_embeddings_tensor = torch.tensor(modern_embeddings)
    serious_embeddings_tensor = torch.tensor(serious_embeddings)
    funny_embeddings_tensor = torch.tensor(funny_embeddings)
    romantic_embeddings_tensor = torch.tensor(romantic_embeddings)
    cynical_embeddings_tensor = torch.tensor(cynical_embeddings)
    rhythmic_embeddings_tensor = torch.tensor(rhythmic_embeddings)
    free_embeddings_tensor = torch.tensor(free_embeddings)
    intense_embeddings_tensor = torch.tensor(intense_embeddings)
    relaxed_embeddings_tensor = torch.tensor(relaxed_embeddings)
    emotional_embeddings_tensor = torch.tensor(emotional_embeddings)
    rational_embeddings_tensor = torch.tensor(rational_embeddings)
    profound_embeddings_tensor = torch.tensor(profound_embeddings)
    superficial_embeddings_tensor = torch.tensor(superficial_embeddings)
    expressive_embeddings_tensor = torch.tensor(expressive_embeddings)
    restrained_embeddings_tensor = torch.tensor(restrained_embeddings)
    happy_embeddings_tensor = torch.tensor(happy_embeddings)
    sad_embeddings_tensor = torch.tensor(sad_embeddings)

    formal_pooled = torch.mean(formal_embeddings_tensor, dim=0) - torch.mean(informal_embeddings_tensor, dim=0)
    traditional_pooled = torch.mean(traditional_embeddings_tensor, dim=0) - torch.mean(modern_embeddings_tensor, dim=0)
    serious_pooled = torch.mean(serious_embeddings_tensor, dim=0) - torch.mean(funny_embeddings_tensor, dim=0)
    romantic_pooled = torch.mean(romantic_embeddings_tensor, dim=0) - torch.mean(cynical_embeddings_tensor, dim=0)
    rhythmic_pooled = torch.mean(rhythmic_embeddings_tensor, dim=0) - torch.mean(free_embeddings_tensor, dim=0)
    intense_pooled = torch.mean(intense_embeddings_tensor, dim=0) - torch.mean(relaxed_embeddings_tensor, dim=0)
    emotional_pooled = torch.mean(emotional_embeddings_tensor, dim=0) - torch.mean(rational_embeddings_tensor, dim=0)
    profound_pooled = torch.mean(profound_embeddings_tensor, dim=0) - torch.mean(superficial_embeddings_tensor, dim=0)
    expressive_pooled = torch.mean(expressive_embeddings_tensor, dim=0) - torch.mean(restrained_embeddings_tensor, dim=0)
    happy_pooled = torch.mean(happy_embeddings_tensor, dim=0) - torch.mean(sad_embeddings_tensor, dim=0)

    torch.save(formal_pooled, 'data/formal_pooled.pt')
    # torch.save(informal_pooled, 'data/informal_pooled.pt')
    torch.save(traditional_pooled, 'data/traditional_pooled.pt')
    # torch.save(modern_pooled, 'data/modern_pooled.pt')
    torch.save(serious_pooled, 'data/serious_pooled.pt')
    # torch.save(funny_pooled, 'data/funny_pooled.pt')
    torch.save(romantic_pooled, 'data/romantic_pooled.pt')
    # torch.save(cynical_pooled, 'data/cynical_pooled.pt')
    torch.save(rhythmic_pooled, 'data/rhythmic_pooled.pt')
    # torch.save(free_pooled, 'data/free_pooled.pt')
    torch.save(intense_pooled, 'data/intense_pooled.pt')
    # torch.save(relaxed_pooled, 'data/relaxed_pooled.pt')
    torch.save(emotional_pooled, 'data/emotional_pooled.pt')
    # torch.save(rational_pooled, 'data/rational_pooled.pt')
    torch.save(profound_pooled, 'data/profound_pooled.pt')
    # torch.save(superficial_pooled, 'data/superficial_pooled.pt')
    torch.save(expressive_pooled, 'data/expressive_pooled.pt')
    # torch.save(restrained_pooled, 'data/restrained_pooled.pt')
    torch.save(happy_pooled, 'data/happy_pooled.pt')
    # torch.save(sad_pooled, 'data/sad_pooled.pt')

formal_pooled = None
informal_pooled = None
traditional_pooled = None
modern_pooled = None
serious_pooled = None
funny_pooled = None
romantic_pooled = None
cynical_pooled = None
rhythmic_pooled = None
free_pooled = None
intense_pooled = None
relaxed_pooled = None
emotional_pooled = None
rational_pooled = None
profound_pooled = None
superficial_pooled = None
expressive_pooled = None
restrained_pooled = None
happy_pooled = None
sad_pooled = None

def load_embeddings():
    global formal_pooled, informal_pooled, traditional_pooled, modern_pooled, serious_pooled, funny_pooled, romantic_pooled, cynical_pooled, rhythmic_pooled, free_pooled, intense_pooled, relaxed_pooled, emotional_pooled, rational_pooled, profound_pooled, superficial_pooled, expressive_pooled, restrained_pooled, happy_pooled, sad_pooled
    #generate_and_save_embeddings()

    if not globals().get('formal_pooled'):
        formal_pooled = torch.load('data/formal_pooled.pt')
    if not globals().get('informal_pooled'):
        informal_pooled = torch.load('data/informal_pooled.pt')
    if not globals().get('traditional_pooled'):
        traditional_pooled = torch.load('data/traditional_pooled.pt')
    if not globals().get('modern_pooled'):
        modern_pooled = torch.load('data/modern_pooled.pt')
    if not globals().get('serious_pooled'):
        serious_pooled = torch.load('data/serious_pooled.pt')
    if not globals().get('funny_pooled'):
        funny_pooled = torch.load('data/funny_pooled.pt')
    if not globals().get('romantic_pooled'):
        romantic_pooled = torch.load('data/romantic_pooled.pt')
    if not globals().get('cynical_pooled'):
        cynical_pooled = torch.load('data/cynical_pooled.pt')
    if not globals().get('rhythmic_pooled'):
        rhythmic_pooled = torch.load('data/rhythmic_pooled.pt')
    if not globals().get('free_pooled'):
        free_pooled = torch.load('data/free_pooled.pt')
    if not globals().get('intense_pooled'):
        intense_pooled = torch.load('data/intense_pooled.pt')
    if not globals().get('relaxed_pooled'):
        relaxed_pooled = torch.load('data/relaxed_pooled.pt')
    if not globals().get('emotional_pooled'):
        emotional_pooled = torch.load('data/emotional_pooled.pt')
    if not globals().get('rational_pooled'):
        rational_pooled = torch.load('data/rational_pooled.pt')
    if not globals().get('profound_pooled'):
        profound_pooled = torch.load('data/profound_pooled.pt')
    if not globals().get('superficial_pooled'):
        superficial_pooled = torch.load('data/superficial_pooled.pt')
    if not globals().get('expressive_pooled'):
        expressive_pooled = torch.load('data/expressive_pooled.pt')
    if not globals().get('restrained_pooled'):
        restrained_pooled = torch.load('data/restrained_pooled.pt')
    if not globals().get('happy_pooled'):
        happy_pooled = torch.load('data/happy_pooled.pt')
    if not globals().get('sad_pooled'):
        sad_pooled = torch.load('data/sad_pooled.pt')

def compare(poem):
    embedded_poem = model.encode(poem)
    # formal_score = cosine_similarity([embedded_poem], [formal_pooled])[0][0] - cosine_similarity([embedded_poem], [informal_pooled])[0][0]
    # traditional_score = cosine_similarity([embedded_poem], [traditional_pooled])[0][0] - cosine_similarity([embedded_poem], [modern_pooled])[0][0]
    # serious_score = cosine_similarity([embedded_poem], [serious_pooled])[0][0] - cosine_similarity([embedded_poem], [funny_pooled])[0][0]
    # romantic_score = cosine_similarity([embedded_poem], [romantic_pooled])[0][0] - cosine_similarity([embedded_poem], [cynical_pooled])[0][0]
    # rhythmic_score = cosine_similarity([embedded_poem], [rhythmic_pooled])[0][0] - cosine_similarity([embedded_poem], [free_pooled])[0][0]
    # intense_score = cosine_similarity([embedded_poem], [intense_pooled])[0][0] - cosine_similarity([embedded_poem], [relaxed_pooled])[0][0]
    # emotional_score = cosine_similarity([embedded_poem], [emotional_pooled])[0][0] - cosine_similarity([embedded_poem], [rational_pooled])[0][0]
    # profound_score = cosine_similarity([embedded_poem], [profound_pooled])[0][0] - cosine_similarity([embedded_poem], [superficial_pooled])[0][0]
    # expressive_score = cosine_similarity([embedded_poem], [expressive_pooled])[0][0] - cosine_similarity([embedded_poem], [restrained_pooled])[0][0]
    # happy_score = cosine_similarity([embedded_poem], [happy_pooled])[0][0] - cosine_similarity([embedded_poem], [sad_pooled])[0][0]
    formal_score = cosine_similarity([embedded_poem], [formal_pooled])[0][0]
    traditional_score = cosine_similarity([embedded_poem], [traditional_pooled])[0][0]
    serious_score = cosine_similarity([embedded_poem], [serious_pooled])[0][0]
    romantic_score = cosine_similarity([embedded_poem], [romantic_pooled])[0][0]
    rhythmic_score = cosine_similarity([embedded_poem], [rhythmic_pooled])[0][0]
    intense_score = cosine_similarity([embedded_poem], [intense_pooled])[0][0]
    emotional_score = cosine_similarity([embedded_poem], [emotional_pooled])[0][0]
    profound_score = cosine_similarity([embedded_poem], [profound_pooled])[0][0]
    expressive_score = cosine_similarity([embedded_poem], [expressive_pooled])[0][0]
    happy_score = cosine_similarity([embedded_poem], [happy_pooled])[0][0]
    
     # Convert scores to NumPy arrays to ensure compatibility with Anvil
    formal_score = np.array(formal_score)
    traditional_score = np.array(traditional_score)
    serious_score = np.array(serious_score)
    romantic_score = np.array(romantic_score)
    rhythmic_score = np.array(rhythmic_score)
    intense_score = np.array(intense_score)
    emotional_score = np.array(emotional_score)
    profound_score = np.array(profound_score)
    expressive_score = np.array(expressive_score)
    happy_score = np.array(happy_score)
    
    return formal_score, traditional_score, serious_score, romantic_score, rhythmic_score, intense_score, emotional_score, profound_score, expressive_score, happy_score
    
    
    return formal_score, traditional_score, serious_score, romantic_score, rhythmic_score, intense_score, emotional_score, profound_score, expressive_score, happy_score

def main():
    # generate_and_save_embeddings()
    load_embeddings()
    print(compare('Sunrise reveals, yet falsehood in its prime,\
With gilded charlatan ray, the eternal mime.\
Alas, dawn breaks in treacherous sublime. \
\
Gold vestures draped upon the worlds grime,\
Damned daybreak, hailed as divine.\
Sunrise reveals, yet falsehood in its prime.'))
    print(compare('Mornings speak of you, each dawn anew,\
Courting shadows away, with golden hue.\
My heart awakes with the sunrise view,\
Feeling loves warmth, but its all from you,\
In you, the sunrise, my heart finds its view'))

if __name__ == "__main__":
    main()