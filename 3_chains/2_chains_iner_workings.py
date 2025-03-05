from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableSequence

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")

prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system","You are facts expert who knows fact about{animal}."),
        ("human","Tell me {count} facts."),
    ]
)

format_prompt = RunnableLambda(lambda x:prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x:model.invoke(x.to_messages())) 
pares_output = RunnableLambda(lambda x:x.content)


chain = RunnableSequence(first=format_prompt, middle=[invoke_model],last = pares_output)

response = chain.invoke({"animal":"cat","count":2})

print(response)