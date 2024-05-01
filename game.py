import pandas as pd
from openai import OpenAI
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import random
from embedding_comparison import load_embeddings, compare

if 'HELICONE_API_KEY' not in os.environ:
    print("You didn't set your Helicone key to the HELICONE_API_KEY env var on the command line.")
    os.environ['HELICONE_API_KEY'] = input("Please input the helicone api key:")

client = OpenAI(base_url="https://oai.hconeai.com/v1", api_key=os.environ['HELICONE_API_KEY'])
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

easy_df = pd.read_csv("data/easy_poems.csv")
medium_df = pd.read_csv("data/medium_poems.csv")
hard_df = pd.read_csv("data/hard_poems.csv")

attributes = ['formal', 'traditional', 'serious', 'romantic', 'rhythmic', 'intense', 'emotional',  'profound', 'expressive', 'happy']

def select_random_poem(poem_df):
  random_index = random.randint(0, len(poem_df) - 1)
  random_poem = poem_df.iloc[random_index].to_dict()
  return (random_poem['Poem Prompt'], random_poem['Poem'], random_poem['Topic'], random_poem['Style'], random_poem['Structure'])

def create_poem_from_prompt(prompt):
  messages = [
      {'role': 'system', 'content': prompt + ". Please keep it within 10 lines and follow the prompt exactly."}
  ]
  response = client.chat.completions.create(
      model='gpt-4',
      messages=messages,
      temperature=1,
      max_tokens=2048,
      top_p=1.0,
      frequency_penalty=0,
      presence_penalty=0
  )
  gpt_response = response.choices[0].message.content
  return gpt_response

def compute_cosine_similarity(prompt1, prompt2):
  prompt1_embedding = model.encode(prompt1)
  prompt2_embedding = model.encode(prompt2)
  similarity = cosine_similarity([prompt1_embedding], [prompt2_embedding])[0][0]
  return similarity

def compute_llm_similarity(poem_1, poem_2):
  prompt = f"Output a score between 0 and 100 for the similarity between these two poems.\nPoem 1: {poem_1}\nPoem 2: {poem_2}"
  messages = [
      {'role': 'system', 'content': prompt}
  ]
  response = client.chat.completions.create(
      model='gpt-4',
      messages=messages,
      temperature=1,
      max_tokens=2048,
      top_p=1.0,
      frequency_penalty=0,
      presence_penalty=0
  )
  gpt_response = response.choices[0].message.content
  try:
    return int(gpt_response) / 100
  except:
    return compute_llm_similarity(poem_1, poem_2)

def llm_output_match(poem, phrase):
  prompt = f'Does the following input mention the attribute {phrase}?\nPoem: {poem}. Respond with only "yes" or "no"'
  messages = [
      {'role': 'system', 'content': prompt}
  ]
  response = client.chat.completions.create(
      model='gpt-4',
      messages=messages,
      temperature=1,
      max_tokens=2048,
      top_p=1.0,
      frequency_penalty=0,
      presence_penalty=0
  )
  gpt_response = response.choices[0].message.content
  return gpt_response

def give_feedback(user_poem, orig_poem_topic, orig_poem_style, orig_poem_structure):
  topic_similarity = llm_output_match(user_poem, orig_poem_topic)
  print(f"Matched topic: {topic_similarity}")
  style_similarity = llm_output_match(user_poem, orig_poem_style)
  print(f"Matched style: {style_similarity}")
  structure_similarity = llm_output_match(user_poem, orig_poem_structure)
  print(f"Matched structure: {structure_similarity}")

def compute_attribute_similarity(orig_poem, user_poem):    
    orig_poem_similarity_scores = compare(orig_poem)
    user_poem_similarity_scores = compare(user_poem)
    diff = 0
    max_diff, max_diff_idx = 0, 0
    for i in range(len(orig_poem_similarity_scores)):
        print(abs(orig_poem_similarity_scores[i] - user_poem_similarity_scores[i]))
        diff += 1 if abs(orig_poem_similarity_scores[i] - user_poem_similarity_scores[i]) > 0.03 else 0
        if abs(orig_poem_similarity_scores[i] - user_poem_similarity_scores[i]) > max_diff:
            max_diff = abs(orig_poem_similarity_scores[i] - user_poem_similarity_scores[i])
            max_diff_idx = i
    return diff, attributes[max_diff_idx]

def compute_final_similarity_and_feedback(user_poem, user_input, orig_poem_topic, orig_poem_style, orig_poem_structure, orig_poem):
    topic_similarity = llm_output_match(user_input, orig_poem_topic)
    print(f"Matched topic: {topic_similarity}")
    style_similarity = llm_output_match(user_input, orig_poem_style)
    print(f"Matched style: {style_similarity}")
    structure_similarity = llm_output_match(user_input, orig_poem_structure)
    print(f"Matched structure: {structure_similarity}")
    diff, diff_attribute = compute_attribute_similarity(user_poem, orig_poem)

    cos_sim = compute_cosine_similarity(user_poem, orig_poem)
    llm_sim = compute_llm_similarity(user_poem, orig_poem)

    print(cos_sim, llm_sim, diff)
    score = ((1 if topic_similarity.lower() == "yes" else 0) + (1 if style_similarity.lower() == "yes" else 0) + (1 if structure_similarity.lower() == "yes" else 0)) * 15 + (10 - diff) * 2 + cos_sim * 20 + llm_sim * 15
    
    missed_true_attribute = ""
    if topic_similarity == "no":
        missed_true_attribute = "You got the topic wrong."
    if style_similarity == "no":
        missed_true_attribute = "You got the style wrong."
    if structure_similarity == "no":
        missed_true_attribute = "You got the structure wrong."

    reaction = ""
    if score > 90:
        reaction = f"Great job! You did an amazing job on this poem."
    elif score > 70:
        reaction = f"Good job! You did a good job on this poem."
    elif score > 50:
        reaction = f"Nice try! You did an okay job on this poem."
    else:
        reaction = f"Try again! You did a bad job on this poem."

    feedback = f"{reaction} {missed_true_attribute} Also, try to make your poem closer to the original poem in terms of the attribute {diff_attribute}!"
    return score, feedback

def main():
    load_embeddings()
    for difficulty in range(3):
        best_score = 0
        difficulty_str = "EASY" if difficulty == 0 else "MEDIUM" if difficulty == 1 else "HARD"
        print(f"DIFFICULTY: {difficulty_str}\n")
        if difficulty == 0:
            orig_prompt, orig_poem, orig_poem_topic, orig_poem_style, orig_poem_structure = select_random_poem(easy_df)
        elif difficulty == 1:
            orig_prompt, orig_poem, orig_poem_topic, orig_poem_style, orig_poem_structure = select_random_poem(medium_df)
        else:
            orig_prompt, orig_poem, orig_poem_topic, orig_poem_style, orig_poem_structure = select_random_poem(hard_df)
        print(f"ORIGINAL POEM\n{orig_poem}\n")
        for round in range(3):
            print(f"ROUND {round+1}")
            user_input = input("Guess the prompt for the poem: ")
            user_poem = create_poem_from_prompt(user_input)
            print(f"USER POEM\n{user_poem}\n")

            poem_similarity, feedback = compute_final_similarity_and_feedback(user_poem, user_input, orig_poem_topic, orig_poem_style, orig_poem_structure, orig_poem)
            print(f"SIMILARITY: {poem_similarity}")
            print(f"FEEDBACK: {feedback}")
            best_score = max(best_score, poem_similarity)

        print(f"BEST SCORE: {best_score}\n\n")
        print(f"The answer was: {orig_prompt}\n")

main()