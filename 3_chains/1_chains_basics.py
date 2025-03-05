from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")

prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system","You are facts expert who knows fact about{animal}."),
        ("human","Tell me {fact_count} facts."),
    ]
)

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({"animal":"tiger","fact_count":1})
print(result)