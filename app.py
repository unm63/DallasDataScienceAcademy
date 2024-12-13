import textwrap

from flask import Flask, render_template, request, send_file, jsonify, url_for
from fpdf import FPDF
import PyPDF2
import docx
import openai
import tempfile
import os
import google.generativeai as genai
import logging
import requests
#import tensorflow as tf
import pinecone
from transformers import pipeline
import google.generativeai as genai
from flask_restful import Resource, Api
from werkzeug.utils import secure_filename
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
import tensorflow_hub as hub
from flask import send_from_directory
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.service_account import Credentials
import json
import markdown
import pandoc
#from fpdf import FPDF
#from bs4 import BeautifulSoup
#from docx import Document
#from docx.shared import Inches
#from weasyprint import HTML
import pdfkit
#import retool
import re



#from pinecone import Pinecone

apikey = input("Type in the Gemini API Key: ")
SCOPES = ['https://www.googleapis.com/auth/drive']
jsonfile = input("Type in name of Json file: ")
creds = Credentials.from_service_account_file(jsonfile, scopes = SCOPES)
#print("data", data)
service = build('drive', 'v3', credentials=creds)


def upload_to_drive(filename, mimetype, parent_folder_id):
    print("working")
    file_metadata = {'name': filename, 'mimeType': mimetype, 'parents': [parent_folder_id]}
    print("still working", file_metadata)
    media = MediaFileUpload(filename, mimetype=mimetype, resumable=True)
    print("STILL WORKING", media)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print("YES", file)
    file_id = file.get('id')
    # Create a permission to make the file publicly accessible
    permission = {
        'type': 'anyone',
        'role': 'reader'
    }
    file_permission = service.permissions().create(fileId=file_id, body=permission).execute()
    print("permission works!")
    # Get the public shareable link
    file = service.files().get(fileId=file_id, fields='webViewLink').execute()
    public_url = file.get('webViewLink')
    print("public url", public_url)
    return public_url
    # Publicly accessible download link




# Replace with your Gemini API key
genai.configure(api_key=apikey)
model = genai.GenerativeModel("gemini-pro")

app = Flask(__name__, static_folder='static')
api = Api(app)

upload_directory = os.getcwd()

#nlp = pipeline("text-generation", model="gpt2")

""""@app.route('/')
def index():
    return render_template('index.html')  # Render"""


@app.route('/upload', methods=['POST'])
def upload():
    print("WORKING")
    document = request.files['document']
    grant_purpose = request.form['grant_purpose']
    target_audience = request.form['target_audience']
    required_funds = request.form['required_funds']
    print("anything")
    try:
        if document:
            print("EXISTS")
            file_path = os.path.join(upload_directory, document.filename)
            document.save(file_path)

            if file_path.endswith('.pdf'):
                with open(file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text = pdf_reader.pages[0].extract_text()
            elif file_path.endswith('.docx'):
                print("It's a doc!")
                doc = docx.Document(file_path)
                text = ''.join([paragraph.text for paragraph in doc.paragraphs])
            else:
                return "Unsupported file format"

            # Process the extracted text and generate grant application
            # Get user input

            # Generate grant application
            generated_text = generate_grant_application(text, grant_purpose, target_audience, required_funds)
            #generated_text = generate_grant_application(text)
            print("Generated text is ", generated_text)
            #return render_template('result.html', generated_text=generated_text)
            #return generated_text
            #return render_template('customize.html', generated_text=generated_text)
            # Instead of rendering a template, use Retool app state
            #retool.state.set('generatedText', generated_text)
            #return jsonify({'generated_text': f"{generated_text.replace('\n', '<br>')}"})
            #return f"{generated_text.replace('\n', '<br>')}"
            #return markdown.markdown(generated_text, output_format = "html")
            #print(markdown.markdown(generated_text))
            return markdown.markdown(generated_text)
            #doc = pandoc.Document()
            #doc.html = html
            #return doc.rtf
            #return "Text extracted and grant application generation initiated."  # Informative message
        else:
            return "No file uploaded"
    except Exception as e:
        logging.error(f"Error processing document: {e}")
        return "An error occurred while processing the document."


@app.route('/download', methods=['POST'])
def download():
    print("Downloading")
    if request.method == 'POST':
        print("posting")
        customized_text = request.form['customized_text']
        soup = BeautifulSoup(customized_text, 'html.parser')
        #customized_text = html_to_plain_text(customized_text)
        #customized_text = BeautifulSoup(customized_text, 'html.parser').get_text()
        #customized_text = request.form['generated_text']
        print("customized text", customized_text)
        file_format = request.form['format']
        print(customized_text)
        print(file_format)
    else:
        return "Invalid request method", 405
    #customized_text = request.args.get('customized_text')
    if not customized_text:
        return "Missing customized text"

    if file_format == 'docx':
        # Generate and download Word document
        #document = Document()
        doc = docx.Document()
        current_paragraph = None

        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'ul', 'ol', 'li', 'strong', 'i']):
            if element.name == 'p':
                current_paragraph = doc.add_paragraph()
                current_paragraph.add_run(element.text)
            elif element.name in ['h1', 'h2', 'h3']:
                current_paragraph = doc.add_paragraph(element.text)
                current_paragraph.style = 'Heading 1'  # Adjust heading style as needed
            elif element.name in ['strong', 'b']:
                if current_paragraph:
                    current_paragraph.add_run(element.text).bold = True
                else:
                    current_paragraph = doc.add_paragraph(element.text)
                    current_paragraph.style = 'Strong'  # Adjust style as needed
            elif element.name == 'i':
                if current_paragraph:
                    current_paragraph.add_run(element.text).italic = True
                else:
                    current_paragraph = doc.add_paragraph(element.text)
                    current_paragraph.style = 'Italic'  # Adjust style as needed
            elif element.name == 'ul' or element.name == 'ol':
                for li in element.find_all('li'):
                    p = doc.add_paragraph(u'\u2022 ' + li.text)
                    current_paragraph = p
        #doc.add_paragraph(customized_text)
        doc.save('generated_grant_application.docx')
        download_url = upload_to_drive('generated_grant_application.docx',
                                       'application/vnd.openxmlformats-officedocument.wordprocessingml.document', '1GKkY0NIUZIglTZg7iUtuqv3iKvxYGFmj')
        #return "C:/Users/ujwal/PyCharmProjects/LLMProject/generated_grant_application.docx"
        #return send_file('generated_grant_application.docx', as_attachment=True)
        #return send_from_directory('generated_grant_application.docx', as_attachment=True)
        return jsonify({'download_url': download_url})
    elif file_format == 'pdf':
        # Generate and download PDF document
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'ul', 'ol', 'li', 'strong']):
            if element.name in ['h1', 'h2', 'h3']:
                pdf.set_font("Arial", size=14 if element.name == 'h1' else 12, style='B')
                pdf.cell(0, 10, element.text, ln=1)
            elif element.name == 'p':
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, element.text)
        pdf.output('generated_grant_application.pdf')
        #HTML(string=customized_text).write_pdf("generated_grant_application.pdf")
        #pdfkit.from_string(customized_text, 'generated_grant_application.pdf')
        print("So far so good!!!")
        # Save the file to the static folder
        #file_path = os.path.join('static', 'generated_document.pdf')  # Replace with appropriate filename
        #pdf.output(file_path)
        #pdfkit.from_string(customized_text, "generated_grant_application.pdf")

        # Return the download URL
        download_url = upload_to_drive('generated_grant_application.pdf', 'application/pdf', '1GKkY0NIUZIglTZg7iUtuqv3iKvxYGFmj')
        return jsonify({'download_url': download_url})
    else:
        return "Invalid format"


def generate_grant_application(text, grant_purpose, target_audience, required_funds):
    # Create the prompt with user input
    prompt = f"""
    Write a grant application based on the following information, excluding this prompt itself:

    Document Text:
    {text}

    Grant Purpose:
    {grant_purpose}

    Target Audience:
    {target_audience}

    Required Funds:
    {required_funds}

    The grant application should include the following sections:
    Title
    1. Introduction
    2. Project Description
    3. Objectives
    4. Methodology
    5. Budget
    6. Evaluation
    7. Conclusion
    

    Ensure the application is well-structured, coherent, and persuasive. I want to see all 7 of the above sections. 
    Nothing should be in bold or surrounded by asterisks apart from the bullet points.
    """

    print(text)
    print(grant_purpose)
    print(target_audience)
    print(required_funds)

    response = model.generate_content(prompt)

    print("Generated!")
    return response.text


if __name__ == '__main__':
    app.run(debug=True)
