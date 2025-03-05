from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant"),
        ("human","Genrate a thank you note for this positive feedback:{feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant"),
        ("human","Genrate a response addressing negative feedback:{feedback}."),
    ]
)

natural_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant"),
        ("human","Genrate a request for more details for this natural feedback:{feedback}."),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant"),
        ("human","Genrate a message to escalate this feedback to a human agent feedback:{feedback}."),
    ]
)

classification_templte = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant"),
        ("human","Classify the sentiment of this feedback as positive,negative,natural,escalate feedback:{feedback}."),  
    ]
)

branches = RunnableBranch(
    (
        lambda x:"positive" in x,
        positive_feedback_template| model |StrOutputParser()
    ),
    (
        lambda x:"negative" in x,
        negative_feedback_template| model |StrOutputParser()
    ),
    (
        lambda x:"natural" in x,
        natural_feedback_template| model |StrOutputParser()
    ),
        escalate_feedback_template| model |StrOutputParser()
)

classification_chain = classification_templte|model|StrOutputParser()
chain = classification_chain|branches

review = "The product is terrible. It broke after just one use and quality very poor."

result = chain.invoke({"feedback":review})

print(result)