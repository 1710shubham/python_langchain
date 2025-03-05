from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")

animal_facts_template=ChatPromptTemplate.from_messages(
    [
        ("system","You are facts expert who knows fact about{animal}."),
        ("human","Tell me {count} facts."),
    ]
)

translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and convert the provide text into {language}."),
        ("human","Translate the following text to {language}:{text}")        
    ]
)

count_words = RunnableLambda(lambda x: f"Word count:{len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output:{"text":output,"language":"french"})

chain = animal_facts_template | model |StrOutputParser()| prepare_for_translation|translation_template|model|StrOutputParser()

result = chain.invoke({"animal":"cat","count":"2"})
print(result)