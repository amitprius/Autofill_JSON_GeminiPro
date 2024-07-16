from flask import Flask, request, jsonify
import fitz  # PyMuPDF for PDF parsing
import docx2txt  # for DOCX parsing
import re
import spacy
#from pymssql._pymssql
import pymssql
import json

import pypandoc
#import easyocr

#from resume_parser import resumeparse
#from pyresparser import ResumeParser

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

import warnings

from PyPDF2 import PdfReader, PdfFileReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Filter out warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Initialize SpaCy for NER
nlp = spacy.load('en_core_web_sm')

# Initialize the EasyOCR reader
#reader = easyocr.Reader(['en'])

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')


# Define the SQL Server connection parameters
SERVER = 'xxx.xxx.xx.xx'
DATABASE = 'xxxxxxx'
USERNAME = 'xxxxx'
PASSWORD = 'xxxxxx'

conn = pymssql.connect(host=SERVER, user=USERNAME, password=PASSWORD, database=DATABASE)
cursor = conn.cursor()
#
# Extract text from .pdf file
#
def extract_text_from_pdf(file_path):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

#
# Extract text from .doc or .docx file
#
def extract_text_from_docx(file_path):
    text = docx2txt.process(file_path)
    return text

#
# Extract text from .odt and .rtf file
#
def extract_text_from_odt_rtf(file_path):
    text = pypandoc.convert_file(file_path)
    return text

#
# Function to extract name using regex
#
def extract_name(text, name_pattern):
    matches = re.findall(name_pattern, text)

    if matches:
        return matches[0]
    return None

#
# Function to extract email using regex
#
def extract_email(text, email_pattern):
    matches = re.findall(email_pattern, text)
    if matches:
        return matches[0]
    return None

#
# Function to extract mobile number using regex
#
def extract_mobile(text, mobile_pattern):
    matches = re.findall(mobile_pattern, text)

    if matches:
        return matches[0]
    return None

#
# Function to extract name using SpaCy NER
#
def extract_name_ner(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            return ent.text
    return None

def extract_skills(text):
    # Tokenize the text
    words = word_tokenize(text)
    # Tag the words with part of speech tags
    pos_tags = pos_tag(words)
    # Create a named entity chunk tree
    named_entities = ne_chunk(pos_tags, binary=False)
    
    person_names = []
    
    # Extract person names
    for subtree in named_entities:
        if isinstance(subtree, Tree) and subtree.label() == 'PERSON':
            name = ' '.join([leaf[0] for leaf in subtree.leaves()])
            person_names.append(name)
    
    return person_names

def is_valid_person_name(name):
    # Basic validation: Check if the name contains alphabets and is not empty
    return bool(name) and any(char.isalpha() for char in name)

# Function to calculate similarity between two names based on single character tokens
def calculate_name_similarity(name1, name2):

    print(name1)
    print(name2)

    tokens1 = set(name1.lower())
    tokens2 = set(name2.lower())
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    similarity_score = len(intersection) / len(union) if union else 0
    return similarity_score

def remove_duplicates(skills):
    seen = set()
    unique_skills = []
    for skill in skills:
        if skill.lower() not in seen:
            unique_skills.append(skill)
            seen.add(skill.lower())

    # Terms related to universities
    university_related_terms = ['university', 'college']  # Add more terms as needed

    # Filter out skills containing university-related terms
    filtered_skills = [skill for skill in unique_skills if not any(term in skill.lower() for term in university_related_terms)]

    return filtered_skills


def get_skill(skills, target_role):

    # Convert resume skills to lowercase for comparison
    resume_skills_lower = [skill.lower() for skill in skills]

    # Write the SQL query to fetch data from the table
    query = 'SELECT ID, EmpRole, Empskills FROM EmployeeSkills'

    # Execute the SQL query
    cursor.execute(query)

    # Fetch all the rows from the executed query
    rows = cursor.fetchall()

    # Update the skills column
    for row in rows:
        id = row[0]
        role = row[1]
        current_skills = json.loads(row[2])
        
        # Only update skills if the role matches the target role
        if role == target_role:
            # Convert current skills to lowercase for comparison
            current_skills_lower = [skill.lower() for skill in current_skills]
            
            # Check if the length of resume skills is greater than or equal to the length of current skills
            if len(skills) >= len(current_skills):
                # Identify new skills to add from resume_skills
                new_skills = [skill for skill in skills if skill.lower() not in current_skills_lower]
                
                if new_skills:
                    # Update skills list with new skills
                    updated_skills = current_skills + new_skills
                    
                    # Update the SQL table with the new skills
                    update_query = 'UPDATE EmployeeSkills SET Empskills = %s WHERE ID = %s'
                    cursor.execute(update_query, (json.dumps(updated_skills), id))
    
    # Commit the transaction
    conn.commit()
    
    # Close the cursor and connection
    cursor.close()
    conn.close()

def divide_name(name):
    try:
        first_name, last_name = name.split()
        return first_name, last_name
    except ValueError:
        # Handle the case where the name does not contain both first and last name
        return name, None
    
def gemini_pro(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

    user_question = "Give me the person name and email"

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    return response['output_text'].split('\n')[0].strip()

    #print("####################### Extract data from LLM - GeminiPro Ends #######################")

@app.route('/api/upload-cv', methods=['POST'])
def upload_cv():

    role = request.form.get('position')
    print("####################### Person Role #######################")
    print(role)

    if 'cvFile' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    cv_file = request.files['cvFile']
    if cv_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    cv_file.save('uploads/cv_file.pdf')  # Save the file to a designated folder

    # Extract text from the CV/Resume file based on file type
    if cv_file.filename.endswith('.pdf'):
        text = extract_text_from_pdf('uploads/cv_file.pdf')

    elif cv_file.filename.endswith('.doc'):
        text = extract_text_from_docx('uploads/cv_file.doc')

    elif cv_file.filename.endswith('.docx'):
        text = extract_text_from_docx('uploads/cv_file.docx')

    elif cv_file.filename.endswith('.odt'):
        text = extract_text_from_odt_rtf('uploads/cv_file.odt')

    elif cv_file.filename.endswith('.rtf'):
        text = extract_text_from_odt_rtf('uploads/cv_file.rtf')

    else:
        return jsonify({'error': 'Unsupported file type'}), 400
 
    
    print("####################### Person name JSON #######################")
    #data = resumeparse.read_file("uploads/cv_file.pdf")
    #name_json = data.get("name", None)

    #data = ResumeParser("resume.pdf").get_extracted_data()
    #name_json = data["name"]

    #name_json = 'Amit'
    #print(name_json)

    # Regex patterns 
    name_pattern = re.compile(r"([A-Z][a-zA-Z\s]+(?:[A-Z][a-zA-Z\s]+)?)")
    email_pattern = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
    mobile_pattern = re.compile(r'(\(\d{3}\)-\d{3}-\d{4}|\d{10}|\+\d{12})')

    # Extracting name and email
    name = extract_name(text, name_pattern)
    email = extract_email(text, email_pattern)
    mobile = extract_mobile(text, mobile_pattern)

    name_json = name
    
    # Alternatively, extract name using SpaCy NER
    name_ner = extract_name_ner(text)
    
    print("####################### Person name text #######################")
    skills = extract_skills(text)
    print(skills[0])
    names = skills[0]
    skills = skills[1:]
    # Remove duplicates
    skills = remove_duplicates(skills)
    print(skills)

    print("####################### Person name Regex #######################")
    print(f"Name (regex): {name}")
    #print(f"Email (regex): {email}")
    print("####################### Person name NER #######################")
    print(f"Name (NER): {name_ner}")
    #print(f"Mobile (regex): {mobile}")

    print("####################### Extract data from LLM - GeminiPro #######################")
    name_llm = gemini_pro(text)
    #name_match = re.search(r'Name:\s*([A-Z\s]+)', name_llm)
    #name_llm = name_match.group(1).strip() if name_match else None

    if skills:
        table_skills = get_skill(skills, 'DataScientist')

    if email:
        # Extract username from email
        username = email.split('@')[0]

        # Remove digits from the username using regex
        username = re.sub(r'\d', '', username)

        # List of detected names
        detected_names = [name_json, skills[0], name, name_ner]
        print(detected_names)
        detected_names = [element for element in detected_names if not isinstance(element, list)]
        

        # Calculate similarity scores
        name_similarity_scores = {}
        for name_i in detected_names:
            if is_valid_person_name(name_i):
                similarity_score = calculate_name_similarity(username, name_i)
                name_similarity_scores[name_i] = similarity_score

        # Find the most related person based on similarity scores
        most_related_person = max(name_similarity_scores, key=name_similarity_scores.get)
        most_related_person = most_related_person.strip()

        print(f"The most related person based on name similarity is: {most_related_person}")

        # Divide the name
        first_name, last_name = divide_name(name_llm)
        print(name_llm)
        #print(last_name)

        # Return extracted data to the frontend
        return jsonify({'first_name': first_name, 'last_name': last_name, 'email': email, 'mobile': mobile, 'skills': skills})
    
    else:
        # Divide the name
        first_name, last_name = divide_name(name)

        return jsonify({'first_name': first_name, 'last_name': last_name, 'email': email, 'mobile': mobile, 'skills': skills})

if __name__ == '__main__':
    app.run(debug=True)
