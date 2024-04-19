
#Name : Tushar 
#Registration number : 23MCA0042
#Branch : MCA
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from ctransformers import AutoModelForCausalLM
#from ctransformers import CTransformers
import sys
DB_FAISS_PATH = "vectorstore/db_faiss"
st.title("CSV Statistical Analysis")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Display data
    st.subheader("Data Preview")
    st.write(data.head())

    # Function to calculate statistical measures
    def calculate_statistics(column):
        if data[column].dtype == 'float64' or data[column].dtype == 'int64':
            mean = data[column].mean()
            median = data[column].median()
            mode = data[column].mode().iloc[0]
            std_dev = data[column].std()
            return mean, median, mode, std_dev
        else:
            return None, None, None, None

    # Function to calculate covariance
    def calculate_covariance(column1, column2):
        if data[column1].dtype == 'float64' and data[column2].dtype == 'float64':
            covariance = data[[column1, column2]].cov().iloc[0, 1]
            return covariance
        else:
            return None

    # Function to generate plots
    def generate_plot(plot_type, x_column, y_column):
        if plot_type == "Histogram":
            if data[x_column].dtype == 'float64' or data[x_column].dtype == 'int64':
                fig, ax = plt.subplots()
                ax.hist(data[x_column])
                st.pyplot(fig)
            else:
                st.error("Selected column does not contain numeric values.")
        elif plot_type == "Scatter Plot":
            if data[x_column].dtype == 'float64' and data[y_column].dtype == 'float64':
                fig, ax = plt.subplots()
                ax.scatter(data[x_column], data[y_column])
                st.pyplot(fig)
            else:
                st.error("Selected columns do not contain numeric values.")
        elif plot_type == "Line Plot":
            if data[x_column].dtype == 'float64' and data[y_column].dtype == 'float64':
                fig, ax = plt.subplots()
                ax.plot(data[x_column], data[y_column])
                st.pyplot(fig)
            else:
                st.error("Selected columns do not contain numeric values.")

    # Sidebar for selecting operation and column
    operation = st.sidebar.selectbox("Select Operation", ["Statistical Measures", "Covariance", "Plots"])

    if operation == "Statistical Measures":
        st.subheader("Select Column and Measure")
        column = st.selectbox("Select Column", data.columns)
        
        mean, median, mode, std_dev = calculate_statistics(column)
        if mean is not None:
            st.write("Column:", column)
            st.write("Mean:", mean)
            st.write("Median:", median)
            st.write("Mode:", mode)
            st.write("Standard Deviation:", std_dev)
        else:
            st.error("Selected column does not contain numeric values.")

    elif operation == "Covariance":
        st.subheader("Calculate Covariance")
        column1 = st.selectbox("Select First Column", data.columns)
        column2 = st.selectbox("Select Second Column", data.columns)
        
        covariance = calculate_covariance(column1, column2)
        if covariance is not None:
            st.write("Covariance between", column1, "and", column2, ":", covariance)
        else:
            st.error("Selected columns do not contain numeric values.")

    elif operation == "Plots":
        st.subheader("Select Plot Type")
        plot_type = st.selectbox("Select Plot Type", ["Histogram", "Scatter Plot", "Line Plot"])
        
        if plot_type == "Histogram" or plot_type == "Scatter Plot" or plot_type == "Line Plot":
            x_column = st.selectbox("Select X-axis Column", data.columns)
            y_column = st.selectbox("Select Y-axis Column", data.columns)
            generate_plot(plot_type, x_column, y_column)

    # Calculate mean, median, mode, standard deviation
    # mean = data.mean()
    # median = data.median()
    # mode = data.mode().iloc[0]
    # std_dev = data.std()

    # st.write("Mean:")
    # st.write(mean)

    # st.write("Median:")
    # st.write(median)

    # st.write("Mode:")
    # st.write(mode)

    # st.write("Standard Deviation:")
    # st.write(std_dev)

    #     # Calculate correlation coefficient
    # st.subheader("Correlation Coefficient")
    # correlation = data.corr()
    # st.write(correlation)

    #     # Generate plots
    # st.subheader("Plots")
    # plot_type = st.selectbox("Select Plot Type", ["Histogram", "Scatter Plot", "Line Plot"])
    # if plot_type == "Histogram":
    #         column = st.selectbox("Select Column for Histogram", data.columns)
    #         fig, ax = plt.subplots()
    #         ax.hist(data[column])
    #         st.pyplot(fig)
    # elif plot_type == "Scatter Plot":
    #         x_column = st.selectbox("Select X-axis Column", data.columns)
    #         y_column = st.selectbox("Select Y-axis Column", data.columns)
    #         fig, ax = plt.subplots()
    #         ax.scatter(data[x_column], data[y_column])
    #         st.pyplot(fig)
    # elif plot_type == "Line Plot":
    #         x_column = st.selectbox("Select X-axis Column", data.columns)
    #         y_column = st.selectbox("Select Y-axis Column", data.columns)
    #         fig, ax = plt.subplots()
    #         ax.plot(data[x_column], data[y_column])
    #         st.pyplot(fig)
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    docsearch = FAISS.from_documents(text_chunks, embeddings)

    docsearch.save_local(DB_FAISS_PATH)

    # hf_hub_download(
    # repo_id="TheBloke/Llama-2-7B-Chat-GGML",
    # filename="llama-2-7b-chat.ggmlv3.q8_0.bin",
    # local_dir="./models")

    # llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML",
    #          model_file="llama-2-7b-chat.ggmlv3.q8_0.bin")
    # qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

    llm = CTransformers(model="./models/llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type="llama",
                        max_new_tokens=512,
                        temperature=0.1)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())
    chat_history=[]
    st.subheader("Ask a Query")
    query = st.text_input("Type your query here")
    button = st.button("Submit")
    if button:
     if query == 'exist':
        st.write('Exiting')
        sys.exit()
     if query == '':
        st.write('Please enter a query')
     else:
        result = qa.invoke({"question": query, "chat_history": chat_history})
        chat_history.append({"question": query, "answer": result['answer']})
        st.write(result['answer'])
     
