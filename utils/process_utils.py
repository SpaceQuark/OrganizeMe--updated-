import os
import docx
import pymupdf
import pandas as pd 
from PIL import Image
import pytesseract
import speech_recognition as sr
import pathlib
 

def process_text_file(file_path, max_chars=5000):
    """
    Read the text from a text file
    """
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read(max_chars)
        return text
    except Exception as e:
        print(f"Error {e} occured when reading the text file: {file_path}!")
        return

def process_pdf_file(file_path, max_chars=5000):
    """
    Read the text from a pdf file
    """
    
    try:
        pdf_document = pymupdf.open(file_path)
        text = []
        tot_chars = 0
        for page in pdf_document:
            page_text = page.get_text()
            tot_chars += len(page_text)
            if tot_chars > max_chars:
                break
            text.append(page_text)
        pdf_document.close()
        return '\n'.join(text)
    except Exception as e:
        print(f"Error {e} occurred while reading the PDF file {file_path}!")
        return


def process_doc_file(file_path, max_chars=5000):
    """
    Read the text from a doc file
    """
    try:
        doc = docx.Document(file_path)
        text = []
        tot_chars = 0
        for paragraph in doc.paragraphs:
            paragraph_text = paragraph.text
            tot_chars += len(paragraph_text)
            if tot_chars > max_chars:
                break
            text.append(paragraph_text)
        return '\n'.join(text)
    except Exception as e:
        print(f"Error {e} occurred while reading the Word document {file_path}!")
        return
    
def process_spreadsheet_file(file_path):
    """
    Process text from a spreadsheet file
    """
    try:
        df = pd.read_csv(file_path) if pathlib.Path(file_path).suffix == ".csv" else pd.read_excel(file_path) 
        return df.to_string(index=False)
        
    except Exception as e:
        print(f"Error {e} occurred while reading the Word document {file_path}!")
        return


def process_image_file(file_path):
    """
    Process text from an image file using OCR
    """
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error {e} occurred while reading the image file {file_path}!")


def process_audio_file(file_path):
    """
    Process text from an audio file using Speech-to-Text (STT)
    """
    try:
        r = sr.Recognizer()

        with sr.AudioFile(file_path) as source:
            audio_data = r.record(source)  

        try:
            text = r.recognize_google(audio_data) 
            return text
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Speech Recognition service; {e}")
            return None

    except FileNotFoundError:
        print(f"Error: Audio file not found at {file_path}!")