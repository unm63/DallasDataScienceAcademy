from flask import Flask, render_template, request, send_file, jsonify
from fpdf import FPDF
import PyPDF2
import docx
import openai
import tempfile
import os
import google.generativeai as genai
import logging
import requests
import tensorflow as tf
import pinecone
from transformers import pipeline
import google.generativeai as genai
from flask_restful import Resource, Api
from werkzeug.utils import secure_filename
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
import tensorflow_hub as hub
from flask import send_from_directory
#from pinecone import Pinecone

# Initialize the Pinecone client
#pinecone = Pinecone()

# Initialize the index
#pinecone.init(
#    api_key="27c95e1e-4a0c-4dd3-b430-60a3a037eabe",
#    environment="us-east-1"
#)

# Get a reference to the index
#index = pinecone.Index("projectindex")

#embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

#def embed_text(text):
#    embeddings = embed([text])
#    return embeddings[0]

# Download and load the pre-trained model weights (complex process)
#model_weights_url = "https://url_to_model_weights.h5"
#model = tf.keras.models.load_model(model_weights_url)

#def embed_text(text):
    # Pre-process and feed text to the model (complex process)
#    embeddings = model.predict(text)
#    return embeddings[0]

#pinecone = Pinecone()
#pc = Pinecone(api_key="27c95e1e-4a0c-4dd3-b430-60a3a037eabe")
#index = pc.Index("quickstart")
#pinecone.init(
#    api_key="27c95e1e-4a0c-4dd3-b430-60a3a037eabe",  # Replace with your Pinecone API key
#    environment="us-east-1"  # Replace with your Pinecone environment
#)

#index_name = "projectindex"
#dimension = 1536  # Adjust based on your embedding model

#index = pinecone.Index(index_name)

#def upload_to_pinecone(text, doc_id):
#    vector = embed_text(text)
#    metadata = {"source": "uploaded_document"}
#    index.upsert(vectors=[vector], ids=[doc_id], metadatas=[metadata])

# Replace with your Hugging Face API key
#HUGGINGFACE_API_KEY = "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"
genai.configure(api_key="AIzaSyBKIplJ67voBYHIlSBbjGCfzppbQLPldTw")
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
    document = request.files['document']
    grant_purpose = request.form['grant_purpose']
    target_audience = request.form['target_audience']
    required_funds = request.form['required_funds']

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
            return jsonify({'generated_text': generated_text})
            #return "Text extracted and grant application generation initiated."  # Informative message
        else:
            return "No file uploaded"
    except Exception as e:
        logging.error(f"Error processing document: {e}")
        return "An error occurred while processing the document."




@app.route('/download', methods=['POST', 'GET'])
def download():
    print("Downloading")
    if request.method == 'GET':
        customized_text = request.args.get('customized_text')
        file_format = request.args.get('format')
        print(customized_text)
        print(file_format)
    elif request.method == 'POST':
        customized_text = request.form['customized_text']
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
        doc = docx.Document()
        doc.add_paragraph(customized_text)
        doc.save('generated_grant_application.docx')
        return send_file('generated_grant_application.docx', as_attachment=True)
        #return send_from_directory('generated_grant_application.docx', as_attachment=True)
    elif file_format == 'pdf':
        # Generate and download PDF document
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, customized_text)
        pdf.output('generated_grant_application.pdf')
        return send_file('generated_grant_application.pdf', as_attachment=True)
        #return send_from_directory('generated_grant_application.docx', as_attachment=True)
    else:
        return "Invalid format"


def generate_grant_application(text, grant_purpose, target_audience, required_funds):
    # Load LLM pipeline (adjust model name as needed)
    #nlp = pipeline("text-generation", model="gpt2")
    # Vectorize the text
    #vector = embed_text(text)

    # Query Pinecone for similar documents
    #query_results = index.query(vector=vector, top_k=5)
    #similar_texts = [result['metadata']['text'] for result in query_results['matches']]
    #prompt1 = f"""Write a grant application with these using {text}, {grant_purpose}, {required_funds}, {target_audience} as inputs"""
    # Create the prompt with user input
    prompt = f"""
    Write a grant application based on the following information, excluding this prompt itself:

    **Document Text:**
    {text}

    **Grant Purpose:**
    {grant_purpose}

    **Target Audience:**
    {target_audience}

    **Required Funds:**
    {required_funds}

    The grant application should include the following sections:
    Title (in bold and in bigger font than the rest)
    1. Introduction
    2. Project Description
    3. Objectives
    4. Methodology
    5. Budget
    6. Evaluation
    7. Conclusion
    
    I want each of the sections to be on a new line, in bold font, and centered on the page. 
    
    **Ensure the following formatting:**

    * **Section Headers:** Bold and centered
    * **Body Text:** Normal font

    Ensure the application is well-structured, coherent, and persuasive. I want to see all 7 of the above sections.
    """

    print(text)
    print(grant_purpose)
    print(target_audience)
    print(required_funds)

    # Set authorization header
    #headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    # Prepare data for LLM (modify prompt if needed)
    #data = {"inputs": prompt, "max_length": 2048}

    # Send request to Hugging Face Inference API
    #response = requests.post(
    #    "https://api-inference.huggingface.co/models/gpt2", headers=headers, json=data
    #)

    # Extract generated summary
    #generated_text = response.json()["generated_texts"][0]
    #response_data = response.json()
    response = model.generate_content(prompt)
    #print(response_data)
    #print("GOOD")
    #print(type(response_data))
    #print(response_data[0])
    #print(response_data[0]["generated_text"])
    #generated_text = response_data[0]["generated_text"]
    #print("generated_text")
    #if "generated_text" in response_data:
    #    print("It's in")
    #    generated_text = response_data["generated_text"]
    #else:
    #    # Handle the case where the "generated_texts" key is missing
    #    logging.error("Missing 'generated_texts' key in LLM response")
    #    return "An error occurred while processing the document."
    print("Generated!")
    return response.text


class UploadDocument(Resource):
    def post(self):
        try:
            print("Function is working!!!!")
            # Get uploaded file and form data
            document = request.files.get('document')
            grant_purpose = request.form['grant_purpose']
            target_audience = request.form['target_audience']
            required_funds = request.form['required_funds']

            if document:
                filename = secure_filename(document.filename)
                filepath = os.path.join(upload_directory, filename)
                document.save(filepath)

                # Process the file based on extension (modify as needed)
                if filepath.endswith('.pdf'):
                    with open(filepath, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        text = pdf_reader.pages[0].extract_text()
                elif filepath.endswith('.docx'):
                    print("It's a doc!")
                    doc = docx.Document(filepath)
                    text = ''.join([paragraph.text for paragraph in doc.paragraphs])
                else:
                    return jsonify({'error': 'Unsupported file format'}), 400

                # Generate grant application
                generated_text = generate_grant_application(text, grant_purpose, target_audience, required_funds)

                # Return JSON response with generated text
                return jsonify({'generated_text': generated_text})
            else:
                return jsonify({'error': 'No file uploaded'}), 400

        except Exception as e:
            logging.error(f"Error processing document: {e}")
            return jsonify({'error': 'An error occurred while processing the document'}), 500


api.add_resource(UploadDocument, '/upload')

if __name__ == '__main__':
    app.run(debug=True)
