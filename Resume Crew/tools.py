import fitz

def read_all_pdf_pages(pdf_path):
    text = ''
    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    return text



import re
from langchain.agents import Tool

def verify_resume(resume_text):
    # Check for missing sections
    required_sections = ['Summary', 'Skills', 'Experience', 'Education']
    for section in required_sections:
        if section not in resume_text:
            return f"Missing '{section}' section in the resume."

    # Check for inconsistent formatting
    section_pattern = r"^[A-Z][a-zA-Z ]*(?=\n|$)"
    experience_pattern = r"^\d{4}-\d{4}\s+(.+)$"
    education_pattern = r"^\d{4}\s+(.+)$"

    for section in resume_text.split('\n'):
        if re.match(section_pattern, section):
            if 'Experience' in section:
                if not re.match(experience_pattern, section):
                    return "Inconsistent formatting in the 'Experience' section."
            elif 'Education' in section:
                if not re.match(education_pattern, section):
                    return "Inconsistent formatting in the 'Education' section."

    # Check for incorrect usage of keywords
    common_keywords = ['team player', 'problem-solving', 'leadership', 'communication', 'adaptability', 'innovation']
    for keyword in common_keywords:
        if keyword not in resume_text:
            return f"Keyword '{keyword}' not found in the resume."

    # If no issues are found, return a success message
    return "Resume verified"

verify_resume_tool = Tool(
    name="Resume Verification",
    func=verify_resume,
    description="Verify the given resume."
)



from langchain.utilities import GoogleSerperAPIWrapper
import os

# Setup API keys
os.environ["OPENAI_API_KEY"] = "Your Key"
os.environ["SERPER_API_KEY"] = "Your Key"

# Set up the Google Search tool
search = GoogleSerperAPIWrapper()

def search_jobs(resume_text):
    # Extract skills and experiences from the resume
    skills = re.findall(r"Skills:\s*(.+)", resume_text, re.MULTILINE)
    experiences = re.findall(r"^\d{4}-\d{4}\s+(.+)$", resume_text, re.MULTILINE)

    # Construct the search query using the extracted skills and experiences
    search_query = f"'{skills[0]}' AND 'experience in {experiences[0]}'"

    # Search for job opportunities using the Google Serper API
    job_search_results = search.run(search_query)

    # Return the search results
    return job_search_results

job_search_tool = Tool(
    name="Job Search",
    func=search_jobs,
    description="Search for job opportunities based on the given resume."
)