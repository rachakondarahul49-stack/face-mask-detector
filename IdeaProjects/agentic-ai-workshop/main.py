from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

response = model.invoke(
    "Explain Artificial Intelligence in two simple sentences."
)

print(response.content)