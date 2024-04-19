# ChatCSV: LangChain LLM and Streamlit Interface with statistical analysis

ChatCSV with Statistical Analysis" is a data-driven chat application that leverages the power of natural language processing (NLP) and statistical analysis techniques to provide insights into conversational data stored in CSV format. Built using Streamlit for the user interface and LangChain for NLP capabilities, this application allows users to upload CSV files containing chat transcripts for in-depth analysis.

- Note - No need any OpenAI API key

## Key Features:

- NLP Processing: Utilizing the LangChain framework, the application processes the textual data from uploaded CSV files, extracting meaningful insights and patterns from conversations.
- Statistical Analysis: Incorporating statistical analysis functions such as mean, median, mode, standard deviation, and covariance, users can gain a deeper understanding of the data's distribution and relationships.
- Visualization Tools: Integrated with Streamlit, the application offers interactive visualization tools, including histograms, scatter plots, and line plots, allowing users to explore and interpret the data visually.

## Get started

1. Clone this GitHub repository to your local machine.
2. Make sure you have Python installed (recommended version is Python 3.7 or higher).
3. Create a python virtual environment using command:
   ```bash
   python -m venv venv
   ```
4. Activate your python environment using command:

   ```bash
   venv\Scripts\activate
   ```

5. Install the required dependencies one by one by running the following command:

   ```bash
   pip install streamlit
   pip install langchain
   pip install langchain_community
   pip install pandas
   pip install matplotlib
   pip install huggingface_hub
   pip install faiss-cpu
   ```

- Note - There may be a need for additional packages such as CTransformers and sentence-transformers. If you encounter any error or warning messages, they will suggest which package you need to download. Simply follow the suggested steps.

6. If pip install huggingface_hub not work then follow this command:
   ```bash
   pip3 install huggingface-hub>=0.17.1
   ```
7. When you run first time `app.py`, uncomment the following code:
   ```bash
    # hf_hub_download(
    # repo_id="TheBloke/Llama-2-7B-Chat-GGML",
    # filename="llama-2-7b-chat.ggmlv3.q8_0.bin",
    # local_dir="./models")
   ```

- This model will be downloaded locally to your project in models directory. The model is nearly 7 GB in size and will take approximately 3 hours to download. Once the download is complete, comment out that code. Otherwise, on the next run, it will start downloading again.

8. Run the application using the following command:

   ```bash
   streamlit run app.py
   ```

9. Access the application by opening your web browser and navigating to `http://localhost:8501`.

# Remember some points

- With time, some packages may change, so you may encounter errors.
- In my project, I have not uploaded the model to GitHub due to its large size, so please download it carefully.
- If there are any doubts, refer to the documentation.

# Contact

If you have any questions or feedback, please don't hesitate to contact me at [haryanatushar@gmail.com](mailto:haryanatushar@gmail.com). I appreciate your interest and support!

Happy chatting with ChatCSV!
