from crewai import Agent
from langchain.agents import Tool
from langchain.utilities import GoogleSerperAPIWrapper

# search_tool = SerperDevTool()
# web_rag_tool = WebsiteSearchTool()

search = GoogleSerperAPIWrapper()

# Create and assign the search tool to an agent
serper_tool = Tool(
  name="Intermediate Answer",
  func=search.run,
  description="Useful for search-based queries",
)

from crewai_tools import ScrapeWebsiteTool

# To enable scrapping any website it finds during it's execution
tool = ScrapeWebsiteTool()
# Create agents which uses these tools

def agents(llm):
    verify_resume = Agent(
        role='Resume reader',
        goal='Verify resumes and identify job opportunities.',
        backstory='An expert in understanding and reviewing any format/design of resume.',
        # tools=[verify_resume_tool],
        verbose=True,
        llm=llm,
        max_iters=1,
        max_rpm=2
    )

    job_researcher = Agent(
        role='Research Analyst',
        goal='Provide up-to-date market analysis of industry job requirements of the domain specified',
        backstory='An expert analyst with a keen eye for market trends.',
        tools=[serper_tool, tool],
        verbose=True,
        llm=llm,
        max_iters=1,
        max_rpm=2
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
        # tools=[modify_resume_tool],
        verbose=True,
        llm=llm,
        max_iters=1,
        allow_delegation=True,
        max_rpm=2
    )

    resume_analyser = Agent(
        role='Resume SWOT Analyser',
        goal=f'Perform a SWOT Analysis on the Resume based on the industry Job Requirements report from job_requirements_researcher and provide a json report.',
        backstory='An expert in hiring so has a great idea on resumes',
        verbose=True,
        llm=llm,
        max_iters=1,
        allow_delegation=True,
        max_rpm=2
    )

    return verify_resume, job_researcher, resume_agent, resume_analyser
