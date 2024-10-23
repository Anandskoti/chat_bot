


from flask import Flask, request,render_template,jsonify
from sentence_transformers import SentenceTransformer
import json
import re
import numpy as np
import os
import io

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def preprocess_text(text):
    chunks = re.split(r"(?=\d{5})", text.strip())
    clean_chunks = [re.sub(r'\n+', ' ', chunk.strip()) for chunk in chunks if chunk.strip()]
    return clean_chunks

def load_and_preprocess(filepath):
    with open(filepath, 'r') as file:
        text = file.read()
    chunks = preprocess_text(text)
    return chunks

def convert_to_vectors(chunks):
    vectors = model.encode(chunks)
    return vectors

def save_vectors_to_json(chunks, vectors, output_json):
    data = [{'chunk': chunk, 'vector': vector.tolist()} for chunk, vector in zip(chunks, vectors)]
    with open(output_json, 'w') as json_file:
        json.dump(data, json_file)

chunks = load_and_preprocess('Engineering.txt')
vectors = convert_to_vectors(chunks)

save_vectors_to_json(chunks, vectors, 'college_data_vectors.json')

from sklearn.metrics.pairwise import cosine_similarity

def load_vectors_from_json(input_json):
    with open(input_json, 'r') as json_file:
        data = json.load(json_file)
    chunks = [item['chunk'] for item in data]
    vectors = [np.array(item['vector']) for item in data]
    return chunks, np.array(vectors)

def find_similar_chunk(query, chunks, vectors, top_n=5):
    query_vector = model.encode([query])[0]

    similarities = cosine_similarity([query_vector], vectors)[0]

    top_indices = similarities.argsort()[-top_n:][::-1]

    similar_chunks = [(chunks[i], similarities[i]) for i in top_indices]
    return similar_chunks

chunks, vectors = load_vectors_from_json('college_data_vectors.json')

import openai

def index():
    return render_template('index.html')

import PyPDF2

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


from PIL import Image
import pytesseract

def read_image(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return text

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        return content

def read_normal_text(file_path):
    return file_path


import docx

def read_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []

    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)


def handle_user_input(file_path,file_type):
    if file_type == '.txt':
        text_text= read_text_file(file_path)
        return text_text
    elif file_type == '.pdf':
        pdf_text = read_pdf(file_path)
        return pdf_text
    elif file_type == '.png':
        image_text = read_image(file_path)
        return image_text
    elif file_type == '.jpg':
        image_text = read_image(file_path)
        return image_text
    elif file_type==None:
         normal_text=read_normal_text(file_path)
         return normal_text
    elif file_type == '.heic':
        image_text = read_image(file_path)
        return image_text
    elif file_type == '.bmp':
        image_text = read_image(file_path)
        return image_text
    elif file_type == '.tif':
        image_text = read_image(file_path)
        return image_text
    elif file_type == '.nef':
        image_text = read_image(file_path)
        return image_text
    elif file_type == '.docx':
        docx_text=read_docx(file_path)
        return docx_text
    else:
          print("unsupported file")



import os

def get_extension(file_path):
    if os.path.isfile(file_path):
        _, ext = os.path.splitext(file_path)

        return ext
    else:
      return None



import openai
import os



app = Flask(__name__)

openai.api_key = "paste a api-key here" 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    conversation_history = [{"role": "system", "content": "we are an expert in entrance exam counseler. Provide students and professionals with detailed information about various types of entrance exams, including college admissions exams, graduate program exams, and professional certification tests. Explain exam formats, study materials, preparation strategies, and important dates. Help users choose the right exam based on their career or academic goals, offer tips for successful preparation, and guide them through the registration process. Answer specific questions related to exam requirements, subjects, and any changes in exam patterns or policies on ly realted to entrance exams give a guidance not give guidance about other this.Provide concise and short answers for all queries. "}]
    
    user_input = request.form['user_input']
    file_type=get_extension(user_input)
    query=handle_user_input(user_input,file_type)
    similar_chunks=find_similar_chunk(query, chunks, vectors)
    query= str(similar_chunks)+query
    conversation_history.append({"role": "user", "content": query})

    try:
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation_history,
                max_tokens=150,
                temperature=0.7
        )

        assistant_response =response['choices'][0]['message']['content']
        conversation_history.append({"role": "assistant", "content": assistant_response})

    except Exception as e:
        assistant_response = f"Error: {str(e)}"

    return render_template('index.html', user_input=user_input, assistant_response=assistant_response)

if __name__ == '__main__':
    app.run(debug=True)
