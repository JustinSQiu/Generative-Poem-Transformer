import pandas as pd
from openai import OpenAI
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import random
import csv

if 'HELICONE_API_KEY' not in os.environ:
    print("You didn't set your Helicone key to the HELICONE_API_KEY env var on the command line.")
    os.environ['HELICONE_API_KEY'] = input("Please input the helicone api key:")

client = OpenAI(base_url="https://oai.hconeai.com/v1", api_key=os.environ['HELICONE_API_KEY'])
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

easy_df = pd.read_csv("data/easy_poems.csv")
medium_df = pd.read_csv("data/medium_poems.csv")
hard_df = pd.read_csv("data/hard_poems.csv")

def select_random_poem(df):
  random_index = random.randint(0, len(df) - 1)
  random_poem = df.iloc[random_index].to_dict()
  return (random_poem['Poem Prompt'], random_poem['Poem'], random_index)

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

def give_feedback(user_poem):
  poem_similarity = llm_output_match(user_poem, df["Topic"][poem_idx])
  print(f"Matched topic: {poem_similarity}")
  poem_similarity = llm_output_match(user_poem, df["Style"][poem_idx])
  print(f"Matched style: {poem_similarity}")
  poem_similarity = llm_output_match(user_poem, df["Structure"][poem_idx])
  print(f"Matched structure: {poem_similarity}")

orig_prompt, orig_poem, poem_idx = select_random_poem(easy_df)
print(f"Original Poem: {orig_poem}")

for i in range(3):
    print("Round {i}")
    user_input = input("Guess the prompt for the poem: ")
    user_poem = create_poem_from_prompt(user_input)
    print(f"User Generated Poem: {user_poem}")

    llm_similarity = compute_llm_similarity(orig_poem, user_poem)
    print(f"LLM computed similarity between original prompt and user prompt: {llm_similarity}")

    poem_similarity = compute_cosine_similarity(orig_poem, user_poem)
    print(f"Cosine similarity between original prompt and user prompt: {poem_similarity}")

    give_feedback(user_poem)

# Set up a separate csv with original poem_idx, first user prompt and improved prompt from user for feedback technique used
# This is used to evaluate the usefulness of each feedback technique
# def write_to_eval_csv(folder_path, eval_method, index, first_prompt, improved_prompt):
#     file_path = os.path.join(folder_path, f'eval_{eval_method}.csv')
#     with open(file_path, 'a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([index, first_prompt, improved_prompt])

# write_to_eval_csv()