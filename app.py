import os
# Warning control
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv, find_dotenv

# Importing necessary libraries
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from IPython.display import Markdown

load_dotenv(find_dotenv())
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

llm = LLM(
    model="ollama/llama3.2",
    base_url="http://host.docker.internal:11434"
)

market_analyst = Agent(
    role="Senior Market Research Analyst",
    goal="study the current market trends, future scenarios, and opportunities by considering the category of a business "
        "company named {company_name} and the types of product it develops.",
    backstory=(
        "Your job as a market research analyst is to look into the markets that the business company named "
        "{company_name} operates, competes in, and report back the market trends, future scenarios, and "
        "opportunities."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

sentiment_analyst = Agent(
    role="Senior Sentiment Analyst",
    goal="Analyze sufficient number of customer feedbacks and responses about the company's products and services "
          "to draw a general conclusion on the internal customer sentiment.",
    backstory="You are a sentiment analyst at {company_name} who is responsible for monitoring the internal customer "
        "sentiment by analyzing what kind of feedbacks and responses customers are giving.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

product_analyst = Agent(
    role="Senior Product Analyst",
    goal="Be a great product metrics analyst to provide insights about the product performance"
         "often comparing various products within the company's portfolio to inform strategic decisions",
    backstory="You are a product analyst at {company_name} who is responsible for tracking internal performance"
              "metrics across the company. You are required to understand how any product is interplaying with other"
              "products within a company.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

research_manager = Agent(
    role="Research Manager",
    goal="Provide a comprehensive report by combining the findings of market research, sentiment analysis, and product "
         "performance to help the business company named {company_name} make informed decisions.",
    backstory="You job as a research manager at {company_name} is to coordinate and combine the work of market analyst, "
               "sentiment analyst, and product analyst to come up with a ultimate business decisions. You are required "
               "to aggregate different insights and professionally. You have to ensure that market analyst, sentiment "
               "analyst, and product analyst are adhering to their instructions and are not deviating from the task. "
               "The individual and final responses must be complete, accurate, and professional.",
    verbose=True,
    llm=llm
)

research_tool = SerperDevTool()

market_research = Task(
    description=(
        "Gather related data by analyzing market trends for {company_name} to provide insights that will eventually help "
        "research manager to make concrete business decisions. Use specialized tools to look into the appropriate "
        "market space for the company. Make sure you do a research about the products the company develops, all of the "
        "possible competitors and their activities to come up with a concrete summary that will be beneficial in making "
        "a report for business decision-making. You can make use of the internet and search engines to do a research "
        "about the company {company_name}."
    ),
    expected_output=(
        "A comprehensive summary on the market research for {company_name} that will be beneficial in making a business "
        "report. The summary must provide a clear picture of the market trends, key findings on products, competitive "
        "analysis, and recommendations. It should also address where the future market is heading and the opportunities "
        "that thecompany can explore."
    ),
    tools=[research_tool],
    agent=market_analyst
)

sentiment_analysis = Task(
    description=(
        "Monitor internal customer feedbacks and responses using forums and community platforms to understand consumer "
        "sentiment and provide insights about the products, services, or overall brand of {company_name}. You must "
        "consider using sufficient data and resources to make a final conclusion. Take into account only the relevant "
        "responses without being biased. Find the best resources to get the most accurate and relevantinformation."
    ),
    expected_output=(
        "A concise summary on the consumer sentiment about the recent activities of the company and its products. The " 
        "summary should include the true opinions of the consumers, what they like and dislike, and the overall "
        "sentiment. It should also outline the recommendations provided by the customers for a specific product. Also, "
        "if necessary, provide a brief comparison and user's say on the similar products from the competitors."
    ),
    tools=[research_tool],
    agent=sentiment_analyst
)

product_analysis = Task(
    description=(
        "Analyze the product performance metrics and provide insights about the product's success and areas for "
        "improvement of the company {company_name}. Make sure you understand how each product interplays with others "
        "within the company. Analyze the given product based on key factors such as features, benefits, target market, "
        "pricing, user reviews, and competitive positioning. Make sure to use everything you have to study about products "
        "and generate a accurate report."
    ),
    expected_output=(
        "Provide a brief summary about the individual product using the product metrics. Your response must display "
        "the quality and quantity of resources you have used to assess the product performance. In addition to that, "
        "also make note on strategic decisions that would manifest after comparing and analyzing the product with "
        "other products within the company."
    ),
    tools=[research_tool],
    agent=product_analyst
)

decision_maker = Task(
    description=(
        "Make a final report for {company_name} by aggregating the findings after verifying their accuracy. The report "
        "should be aggregation of the insights provided by the market analyst, sentiment analyst, and product analyst. "
        "Review the individual responses and make sure they are not going off-topic. Check for references and "
        "sources used to find the information, ensuring they are credible and relevant."
    ),
    expected_output=(
        "A final, detailed and informative report ready to be used by business decision-makers. The report should "
        "address the market space, internal customer sentiments, their feedbacks, and internal performance metrics "
        "properly. It should be professional and contain a ultimate business decisions that will help "
        "the company's growth."
    ),
    agent=research_manager,
    output_file="report.md"
)

crew = Crew(
    agents=[market_analyst, sentiment_analyst, product_analyst, research_manager],
    tasks=[market_research, sentiment_analysis, product_analysis, decision_maker],
    verbose=True,
    memory=True,
    embedder=dict(
        provider="ollama",
        config=dict(
            model="ollama/nomic-embed-text",
            base_url="http://host.docker.internal:11434",
        )
    )
)

inputs = {
    "company_name": "Salesforce"
}

report = crew.kickoff(inputs=inputs)
Markdown(report.raw)
