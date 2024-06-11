import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import os

# Function to read text from a PDF file
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")  # Read the PDF file content as a stream
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to read text from a TXT file
def read_txt(file):
    try:
        text = file.read().decode('utf-8')  # Read the content of the UploadedFile as text
        return text
    except Exception as e:
        st.error(f"Error reading TXT file: {e}")
        return ""

# Function to read text from a CSV file with error handling
def read_csv(file):
    try:
        df = pd.read_csv(file, on_bad_lines = 'warn')
    except pd.errors.ParserError:
        st.error("Error parsing CSV file. Please check the file for inconsistencies.")
        return ""
    text = " ".join(df.astype(str).apply(" ".join, axis=1))
    return text

# Main function to handle file uploads and display results
def main():
    st.title("Document Similarity Checker")

    # File upload widgets
    file1 = st.file_uploader("Upload the first file", type=["pdf", "txt", "csv"])
    file2 = st.file_uploader("Upload the second file", type=["pdf", "txt", "csv"])

    if file1 is not None and file2 is not None:
        ext1 = os.path.splitext(file1.name)[1].lower()
        ext2 = os.path.splitext(file2.name)[1].lower()

        # Read contents based on file type
        if ext1 == ".pdf":
            text1 = read_pdf(file1)
        elif ext1 == ".txt":
            text1 = read_txt(file1)
        elif ext1 == ".csv":
            text1 = read_csv(file1)

        if ext2 == ".pdf":
            text2 = read_pdf(file2)
        elif ext2 == ".txt":
            text2 = read_txt(file2)
        elif ext2 == ".csv":
            text2 = read_csv(file2)

        # Ensure text is not empty before processing
        if text1 and text2:
            # Calculate similarity
            cv = CountVectorizer()
            vector = cv.fit_transform([text1, text2])
            similarity = cosine_similarity(vector)[0, 1]
            similarity_percentage = round(similarity * 100, 2)

            # Display results
            st.write(f"**Similarity: {similarity_percentage}%**")
            st.write("Count Vectorizer DataFrame:")
            st.dataframe(pd.DataFrame(vector.toarray(), columns=cv.get_feature_names_out()))
        else:
            st.error("Error reading one or both of the files. Please check the files and try again.")

if __name__ == "__main__":
    main()

