from crewai import Agent, Crew, Process, Task
# from crewai_tools import SerperDevTool, WebsiteSearchTool
from tools import *
# from crewai_tools import tool, DirectoryReadTool, FileReadTool, SerperDevTool, WebsiteSearchTool
from dotenv import load_dotenv, find_dotenv # Groq, Serper
from langchain_groq import ChatGroq 
import streamlit as st
import os
import fitz

load_dotenv(find_dotenv())

# Configuration
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Load the llm
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

# Streamlit UI for uploading the resume
st.title("Job Search Crew")
st.subheader("Upload your resume and enter the desired job")

resume_file = st.file_uploader("Choose a PDF file", type=["pdf"])
job_desire = st.text_input("Enter Desiring Job: ")
Experience= st.text_input("Enter your Experince: ")

if resume_file is not None:
    # Read the uploaded PDF file and convert it to text using fitz
    pdf_doc = fitz.open(resume_file)
    resume_content = ""
    for page in range(pdf_doc.page_count):
        page_content = pdf_doc[page].get_text("text")
        resume_content += page_content

    # Create agents
    def agents(llm):
        verify_resume = Agent(
            role='Resume reader',
            goal='Verify resumes and identify job opportunities.',
            backstory='An expert in understanding and reviewing any format/design of resume.',
            tools=[verify_resume_tool],
            verbose=True,
            llm=llm,
            max_iters=1
        )

        job_researcher = Agent(
            role='Research Analyst',
            goal='Provide up-to-date market analysis of industry job requirements of the domain specified',
            backstory='An expert analyst with a keen eye for market trends.',
            tools=[search_tool, web_rag_tool],
            verbose=True,
            llm=llm,
            max_iters=1
        )

        resume_agent = Agent(
            role="Resume Writer",
            goal="Tailor resumes to specific job requirements and optimize them for better job search outcomes.",
            backstory="""
                You are an experienced resume writer with a passion for helping job seekers stand out in a competitive job market. 
                With your keen eye for detail and a deep understanding of the latest resume writing trends, you have a proven track record of 
                helping individuals land their dream jobs. You specialize in crafting compelling resumes that showcase an individual's unique skills and experiences, making them stand out from the crowd.
                As a resume writer, you are proficient in identifying key skills and experiences that align with specific job requirements. 
                You have an extensive knowledge of various industries and job roles, enabling you to tailor resumes effectively. 
                You stay up-to-date with the latest resume writing techniques and best practices, ensuring that your clients have the best 
                possible chance of success in their job search.
                You are known for your ability to create resumes that are both visually appealing and informative. 
                You understand the importance of using clear and concise language, as well as incorporating relevant keywords and phrases that will catch the attention of recruiters and hiring managers. 
                Your goal is to help job seekers present themselves in the best possible light, increasing their chances of securing interviews and ultimately, their dream jobs.
                Your mission as a Resume Agent is to leverage your expertise in resume writing and your understanding of the job market to modify resumes based on specific job requirements. 
                By tailoring resumes to match the needs of employers, you will help job seekers stand out from the competition and achieve their career goals.""",
            tools=[modify_resume_tool],
            verbose=True,
            llm=llm,
            max_iters=1,
            allow_delegation=True
        )

        resume_analyser = Agent(
            role='Resume SWOT Analyser',
            goal=f'Perform a SWOT Analysis on the Resume based on the industry Job Requirements report from job_requirements_researcher and provide a json report.',
            backstory='An expert in hiring so has a great idea on resumes',
            verbose=True,
            llm=llm,
            max_iters=1,
            allow_delegation=True
        )

        return verify_resume, job_researcher, resume_agent, resume_analyser

    # Create tasks
    def tasks(llm, job_desire, resume_content):
        '''
        job__research - Find the relevant skills, projects and experience
        resume_analysis- understand the report and the resume based on this make a swot analysis
        '''

        verify_resume, job_researcher, resume_agent, resume_analyser  = agents(llm)

        research_task = Task(
            description=f'For Job Position of Desire: {job_desire} research to identify the current market requirements for a person at the job including the relevant skills, some unique research projects or common projects along with what experience would be required. For searching query use ACTION INPUT KEY as "search_query"',
            expected_output='A report on what are the skills required and some unique real time projects that can be there which enhances the chance of a person to get a job',
            agent=job_researcher
        )

        resume_analysis_task = Task(
            description=f'Resume Content: {resume_content} \n Analyse the resume provided and the report of job_requirements_researcher to provide a detailed SWOT analysis report on the resume along with the Resume Match Percentage and Suggestions to improve',
            expected_output='A JSON formatted report as follows: "candidate": candidate, "strengths":[strengths], "weaknesses":[weaknesses], "opportunities":[opportunities], "threats":[threats], "resume_match_percentage": resume_match_percentage, "suggestions": "suggestions"',
            agent=resume_analyser,
            output_file='resume-report/resume_review.json'
        )

        # Task for verify_resume agent
        verify_resume_task = Task(
            description='Verify the resume\'s format, design, and content to ensure it meets industry standards and is easy to read.',
            expected_output='A boolean value indicating whether the resume is verified (True) or not (False).',
            agent=verify_resume
        )

        # Tasks for resume_agent
        analyze_requirements_task = Task(
            description='Analyze the job requirements report generated by the job_researcher agent to identify the key skills, projects, and experience required for the desired job.',
            expected_output='A list of the most important skills, projects, and experience required for the desired job.',
            agent=resume_agent
        )

        modify_resume_task = Task(
            description='Modify the resume to align with the identified key skills, projects, and experience.',
            expected_output='A modified version of the resume that highlights the key skills, projects, and experience required for the desired job.',
            agent=resume_agent
        )

        return verify_resume_task, research_task, analyze_requirements_task, modify_resume_task, resume_analysis_task

    # Assemble a crew
    crew = Crew(
        agents=[verify_resume, job_researcher, resume_agent, resume_analyser],
        tasks=[verify_resume_task, research_task, analyze_requirements_task, modify_resume_task, resume_analysis_task],
        verbose=1,
        process=Process.sequential
    )

        # Kick off the crew
    result = crew.kickoff()

        # Display the output of the agents in Streamlit
    st.subheader("Agent Output")
    st.write(result)