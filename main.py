from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,Settings,StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
import streamlit as st
from llama_index.llms.gemini import Gemini
from llama_parse import LlamaParse
from chromadb.config import Settings as chroma_settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate as PT
from langchain_core.prompts import PromptTemplate
import chromadb
import json
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os

load_dotenv() 
gemini_api_key=os.getenv("GEMINI_API_KEY")
llama_parse_api_key=os.getenv("LLAMA_PARSE_API_KEY")



def load_data(files):
    parser=LlamaParse(api_key=llama_parse_api_key,
                      result_type='markdown',
                      encoding='utf-8')
    file_extractor = {
                        ".docx": parser,
                        ".pdf" : parser,
                        ".txt" : parser,
                        ".md" : parser,
                        "doc" : parser
                        }
    
    documents=SimpleDirectoryReader(input_files=[files],file_extractor=file_extractor).load_data()
    
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    llm = Gemini(
    model="models/gemini-1.5-flash",
    api_key=gemini_api_key,  # uses GOOGLE_API_KEY env var by default
)
    Settings.embed_model = embed_model
    Settings.llm = llm

    chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=chroma_settings(persist_directory='./db')
        )
    collection_name = 'c-2'
    collection_exists=False

    if not chroma_client:
            raise ValueError("ChromaDB client not initialized. Call parse_file first.")
            
    existing_collections = chroma_client.list_collections()
    
        
    for collection in existing_collections:
        if collection == collection_name:
            # chroma_client.delete_collection(collection_name)
            chroma_collection=chroma_client.get_collection(collection)
            collection_exists=True
            print(f"Collection '{collection_name}' deleted.")
            break

    if not collection_exists:
        
        chroma_collection = chroma_client.create_collection(name=collection_name)

        # Set up vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index and query engine

        index = VectorStoreIndex(documents, storage_context=storage_context)
    else:
        # Set up vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index and query engine

        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

    query_engine = index.as_query_engine()
    return query_engine,documents


def summarize(query_engine,question):
    qa_prompt_temp_str="""
                            **You are an AI assistant specializing in document analysis. Given the following document, generate a concise summary capturing its main points, key insights, and essential details.**
                             **Ensure the summary retains the core meaning while being clear and informative. If the document is technical or academic, highlight its main arguments, conclusions, and any critical data points. **
                             **If the document is lengthy, provide a structured summary with bullet points or sections for better readability."**
                             **Provide in a concise way with proper titles and bullet points.**
                             **Also give like a student can have quick recap from it**
                            "Context information is below.\n"
                            "---------------------\n"
                            "{context_str}\n"
                            "---------------------\n"
                            "Answer: "                           
                            """
        # print("RAW PROMPT: ",qa_prompt_temp_str)
        # print()
    qa_prompt_temp = PT(qa_prompt_temp_str)
        # print("QA_PROMPT_TEMPLte: ",qa_prompt_temp)
    query_engine.update_prompts({"response_synthesizer:text_qa_templete":qa_prompt_temp})
        
    response = query_engine.query(question)
    print("response:-",response)
    return response

def process_query(query_engine,question):
    # qa_prompt_temp_str = f"""Question: {question} if answer of this question are in this index(vector_database) then only provide me answer othervise say "not provided in file" """
    qa_prompt_temp_str="""
                            **Strictly give answer from the context provided below and follow rules**
                            "Context information is below.\n"
                            "---------------------\n"
                            "{context_str}\n"
                            "---------------------\n"
                            "Rules: "
                                -"**Use only the information from the provided documents to answer the query.** "\n"
                                -"**Do not generate or infer information outside of the context. If the information is not present in the documents, reply with "Out of context.**"
                                -"**If it is an empty string then say please provide question **"
                                -"**Avoid any form of hallucination or assumptions. Provide answers that are directly supported by the context.**"
                            
                            *Here is the query*
                            "Query: {query_str}\n"
                            "Answer: "                           
                            """
        # print("RAW PROMPT: ",qa_prompt_temp_str)
        # print()
    qa_prompt_temp = PT(qa_prompt_temp_str)
        # print("QA_PROMPT_TEMPLte: ",qa_prompt_temp)
    query_engine.update_prompts({"response_synthesizer:text_qa_templete":qa_prompt_temp})
        
    response = query_engine.query(question)
    print("response:-",response)
    return response

def generate_questions(context):
     print("generating questions: ")
     llm = Gemini(
    model="models/gemini-1.5-flash",
    api_key="AIzaSyDQwxmLu33CfutC6PazlDp8k8xP1T-D_QM",  # uses GOOGLE_API_KEY env var by default
)
     template_string='''
        Generate 3 Multiple choice type questions with 4 options and correct answer from provided context only.
        context: {context}\n
        output format:a list of dictionary with question as key and correct answer as answer and options with list of four different options. .
       Make sure you provide strictly in list format.
            '''
     prompt = PromptTemplate(
            template=template_string,
            input_variables=['context'],
        )
     _input = prompt.format_prompt(context=context)
     output = llm.complete(_input.to_string())
     print("output of generating questions" )
     print("raw output ",output)
     output=str(output)
     output=output.strip("```json").strip("\n```").strip()
     output=json.loads(output)
     print(type(output))
     return output


# input_data=['D:\\Personal_Projects\\IntelliScan\\docs\\sample_pdf-3.pdf','D:\\Personal_Projects\\IntelliScan\\docs\\sample_pdf-2.pdf']
# query_engine,documents=load_data(input_data)
# while True:
#     question=input("Ask question or type quit or press 1 for mcq: ")
#     if question=='quit':
#         break
#     elif question==str(1):
#          op=generate_questions(documents)
#     else:
#         answer=process_query(query_engine,question)
def reset_quiz():
    st.session_state.score = 0
    st.session_state.current_question = 0
    st.session_state.submitted = False
def run():
     if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
     if 'extracted_texts' not in st.session_state:
            st.session_state.extracted_texts = {}
     if 'vector_stores' not in st.session_state:
            st.session_state.vector_stores = {}
     if "messages" not in st.session_state:
        st.session_state.messages = []
     if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None
        st.session_state.documents = None
     if 'questions' not in st.session_state:
          st.session_state.questions=[]
     if 'user_response' not in st.session_state:
          st.session_state.user_response=[]
     if 'score' not in st.session_state:
          st.session_state.score = 0
     if 'current_question' not in st.session_state:
          st.session_state.current_question = 0
     if 'submitted' not in st.session_state:
          st.session_state.submitted = False
     if "selected_answer" not in st.session_state:
          st.session_state.selected_answer = None
     
     
     """Main Streamlit application"""
     st.title("📄 Document Intelligence Assistant")
    # File uploader
     uploaded_file = st.file_uploader("Upload PDF, DOCX, or TXT file", 
                                         type=['pdf', 'docx', 'txt'])
        
     if uploaded_file is not None:
            # Extract text
         st.session_state.uploaded_file = uploaded_file
         if st.session_state.query_engine is None:
            # Save uploaded file
            file_path = 'file.pdf'
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getvalue())
            
            # Process file only once
            with st.spinner("Processing your file.."):
                query_engine, documents = load_data(file_path)
                st.success("File processed successfully!")
                # Store in session state
            st.session_state.query_engine = query_engine
            st.session_state.documents = documents
                
        #  st.session_state.extracted_text = self.extract_text_from_file(uploaded_file)
            
        #     # Create vector store
        #  st.session_state.vector_store = self.create_vector_store(st.session_state.extracted_text)
            
            # Buttons for different actions
         action = st.radio("Choose an action:", 
                              ["Summarize", "Ask Questions", "Take Quiz"],index=None)
            
         if action == "Ask Questions":
             st.write("📋Done processing:")
              # Display chat messages from history
             for message in st.session_state.messages:
                 with st.chat_message(message["role"]):
                     st.markdown(message["content"])
            
            # User input
             if prompt := st.chat_input("Enter your message"):
                 # Display user message in chat container
                 st.chat_message("user").markdown(prompt)
                
                # Add user message to chat history
                 st.session_state.messages.append({
                    "role": "user", 
                    "content": prompt
                 } )
                
                # Generate AI response
                 response = process_query(st.session_state.query_engine,prompt)
                
                # Display AI response
                 with st.chat_message("assistant"):
                     st.markdown(response)
                
                # Add AI response to chat history
                 st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                 })
             
            
         elif action == "Summarize":
             st.markdown("Conclusion: ")
             ans=summarize(st.session_state.query_engine,"Summerize the document ")
             st.markdown(ans)
              
            #  if st.session_state.vector_store:
            #      self.ask_questions(st.session_state.vector_store)
            #  else:
            #     st.warning("Vector store not created. Please re-upload the document.")
            
        #  elif action == "Generate Mind Map":
        #      st.write("mindmaps")
            
         elif action == "Take Quiz":
            st.write("take quiz")
            st.session_state.user={}
            questions=[]
            answers=[]
            options=[]
            
            mcq_data=generate_questions(st.session_state.documents)
            # for d in mcq_data:
            #     questions.append(d["question"])
            #     answers.append(d["answer"])
            #     options.append(["options"])       
            st.session_state.questions=questions
            if st.session_state.current_question < len(mcq_data):
                current_mcq = mcq_data[st.session_state.current_question]
            
                # Display progress
                st.progress((st.session_state.current_question) / len(mcq_data))
                st.write(f"Question {st.session_state.current_question + 1} of {len(mcq_data)}")
                
                # Display question
                st.subheader(current_mcq["question"])
                
                # Create radio buttons for options
                user_answer = st.radio(
                        "Choose your answer:",
                        current_mcq["options"],
                        index=None,
                        key=f"q_{st.session_state.current_question}"
                    )
                #  Submit button
                if st.button("Submit Answer"):
                    if user_answer == current_mcq["answer"]:
                        st.success("Correct! 🎉")
                        st.session_state.score += 1
                    else:
                        st.error(f"Wrong! The correct answer is {current_mcq['answer']} 😔")
                    
                    # Move to next question
                    st.session_state.current_question += 1
                    st.rerun()
            else:
                # Quiz completed - show results
                st.success("Quiz Completed! 🎉")
                st.write(f"Your final score: {st.session_state.score}/{len(mcq_data)}")
            
                # Calculate percentage
                percentage = (st.session_state.score / len(mcq_data)) * 100
                st.write(f"Percentage: {percentage:.2f}%")
            
                # Provide feedback based on score
                if percentage >= 80:
                    st.balloons()
                    st.write("Excellent performance! 🌟")
                elif percentage >= 60:
                    st.write("Good job! Keep practicing! 👍")
                else:
                    st.write("You might want to study more. Keep trying! 📚")
                
                # Restart button
                if st.button("Restart Quiz"):
                    reset_quiz()
                    st.rerun()
            #  print()
            # #  print("op: ",st.session_state.questions[0])
            #  for current_question in op:
            #       print("question: ",current_question)
            #       st.write("Question: ",current_question["question"] )
            #       user_answer = st.radio(
            #             "Select your answer:", 
            #             current_question['options'], 
            #             key=f"q{st.session_state.current_question}"
            #         )
            #       st.session_state.user["question"]=current_question["question"]
            #       st.session_state.user["answer"]=user_answer
            #       st.session_state.user_response.append(st.session_state.user)


run()
