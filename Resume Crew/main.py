import os
from crewai import Crew, Process
# from tools import *
from agents import agents
from task import tasks
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain_groq import ChatGroq
import streamlit as st
import os
import fitz


# Configuration
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# Load the llm
llm = ChatGroq(model="llama3-8b-8192", temperature=0)

# Streamlit UI for uploading the resume
st.title("Job Search Crew")
st.subheader("Upload your resume and enter the desired job")

resume_file = st.file_uploader("Choose a PDF file", type=["pdf"])
job_desire = st.text_input("Enter Desiring Job: ")
Experience= st.text_input("Enter your Experince: ")

if resume_file is not None:
    # Read the uploaded PDF file and convert it to text using fitz
    pdf_doc = fitz.open(resume_file)
    resume = ""
    for page in range(pdf_doc.page_count):
        page_content = pdf_doc[page].get_text("text")
        resume += page_content


# Creating agents and tasks
verify_resume, job_researcher, resume_agent, resume_analyser = agents(llm)

verify_resume_task, research_task, analyze_requirements_task, modify_resume_task, resume_analysis_task = tasks(llm, job_desire, resume_file, Experience)

crew = Crew(
    agents=[verify_resume, job_researcher, resume_agent, resume_analyser],
    tasks=[verify_resume_task, research_task, analyze_requirements_task, modify_resume_task, resume_analysis_task],
    verbose=1,
    process=Process.sequential
)

result = crew.kickoff()
print(result)

st.subheader("Agent Output")
st.write(result)

