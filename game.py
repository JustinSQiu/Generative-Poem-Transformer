import pandas as pd
from openai import OpenAI
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import random

if 'HELICONE_API_KEY' not in os.environ:
    print("You didn't set your Helicone key to the HELICONE_API_KEY env var on the command line.")
    os.environ['HELICONE_API_KEY'] = input("Please input the helicone api key:")

client = OpenAI(base_url="https://oai.hconeai.com/v1", api_key=os.environ['HELICONE_API_KEY'])
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

easy_df = pd.read_csv("data/easy_poems.csv")
medium_df = pd.read_csv("data/medium_poems.csv")
hard_df = pd.read_csv("data/hard_poems.csv")

def select_random_poem(poem_df):
  random_index = random.randint(0, len(poem_df) - 1)
  random_poem = poem_df.iloc[random_index].to_dict()
  return (random_poem['Poem Prompt'], random_poem['Poem'], random_poem['Topic'], random_poem['Style'], random_poem['Structure'])

def create_poem_from_prompt(prompt):
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
  return int(gpt_response) / 100

def llm_output_match(poem, phrase):
  prompt = f'Does the following poem have the attribute {phrase}?\nPoem: {poem}. Respond with only "yes" or "no"'
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

for difficulty in range(3):
    print(f"Difficulty {difficulty}")
    if difficulty == 1:
       orig_prompt, orig_poem, orig_poem_topic, orig_poem_style, orig_poem_structure = select_random_poem(easy_df)
    elif difficulty == 2:
         orig_prompt, orig_poem, orig_poem_topic, orig_poem_style, orig_poem_structure = select_random_poem(medium_df)
    else:
        orig_prompt, orig_poem, orig_poem_topic, orig_poem_style, orig_poem_structure = select_random_poem(hard_df)
    print(f"Original Poem: {orig_poem}")
    for round in range(3):
        print(f"Round {round}")
        user_input = input("Guess the prompt for the poem: ")
        user_poem = create_poem_from_prompt(user_input)
        print(f"User Generated Poem: {user_poem}")

        # llm_similarity = compute_llm_similarity(orig_poem, user_poem)
        # print(f"LLM computed similarity: {llm_similarity}")

        poem_similarity = compute_cosine_similarity(orig_poem, user_poem)
        print(f"Cosine similarity: {poem_similarity}")

        give_feedback(user_poem, orig_poem_topic, orig_poem_style, orig_poem_structure)