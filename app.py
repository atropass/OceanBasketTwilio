from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import openai
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json 
# Initialize the Flask app
app = Flask(__name__)
load_dotenv()

# Load the OpenAI API key from an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
ZILLIZ_URI=os.getenv("ZILLIZ_URI")
ZILLIZ_USER=os.getenv("ZILLIZ_USER")
ZILLIZ_PASSWORD=os.getenv("ZILLIZ_PASSWORD")

def get_db():
    embeddings = OpenAIEmbeddings()
    database = Milvus(
        embedding_function=embeddings,
        collection_name="Answers",
        connection_args={
            "uri": ZILLIZ_URI,
            "user": ZILLIZ_USER,
            "password": ZILLIZ_PASSWORD,
            "secure": True,
        },
        drop_old=False
    )
    print("Success Zilliz Connection")
    return database

conversation_histories = {}

def add_message_to_history(user_id, sender, message):
    now = datetime.now()
    if user_id not in conversation_histories:
        conversation_histories[user_id] = []
    conversation_histories[user_id].append((now, f"{sender}: {message}"))

def get_recent_history(user_id, hours=6):
    if user_id not in conversation_histories:
        return []
    
    now = datetime.now()
    cutoff_time = now - timedelta(hours=hours)
    return [msg for time, msg in conversation_histories[user_id] if time > cutoff_time]
    
def generate_answer(query, user_id):
    db = get_db()
    docs = db.similarity_search(query, k=5)
    docs_page_content = "".join([d.page_content for d in docs])
    
    recent_history = get_recent_history(user_id, 6)
    print(recent_history)
    history_str = "\n".join(recent_history)
    print("\n=============================\n")
    print("history_str:", history_str)
    print("\n=============================\n")

    prompt = f"""
    You are a bot assistant of Ocean Basket Kazakhstan working in the field of seafood restaurants. Your tone should be friendly and sociable. You help to book a table, order delivery, take complaints, reviews, and suggestions. 
    
    RETURN STRICTLY text in the following format and use double quotes STRICTLY:
    {{
        "type" : "options : order, comlaint, review, suggestion",
        "content" : "latest query",
        "summary" : "summary of what the discussion is, include all the neccessary details related to type of the discussion",
        "status" : "status of the discussion, if the order, complaint, review, suggestion is clear and client have finished set the status using these options: done, processing.",
        "message_to" : "message to client",
    }}


    Classify the message as either an "Order", "Complaint", "Review", or "Suggestion".
    
    Respond in the language used by the customer, either Kazakh, Russian, or English. All information about Ocean Basket is available in {docs_page_content}.
    
    Maintain the context of the conversation based on the history provided.

    Example of a dialogue in case of an order:
    ChatGPT: Здравствуйте! Спасибо, что обратились в Ocean Basket. Как я могу помочь вам сегодня?
    Customer: Я хотел бы сделать заказ.
    ChatGPT: Отлично! Пожалуйста, укажите, что вы хотели бы заказать.
    Customer: [Описание заказа]
    ...

    Do not finish the order discussion until you get the all necessary information from the customer, like the name, phone number, address, and order information.
    
    Conversation History:
    {history_str}
    
    Latest Query: {query}
    Response and Classification:"""

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        max_tokens=450
    )
    full_response = response.choices[0].message["content"]
    print("\n===========================FULL RESPONSE======================\n")
    print(full_response)
    dic = json.loads(full_response)
    print("\n===========================DICTIONARY======================\n")
    print(dic)
    print("\n===========================================================\n")


    return dic["message_to"]


# Define a route to handle incoming requests
@app.route("/chatgpt", methods=["POST"])
def chatgpt():
    # try:
    print(request.values)
    wa_id = request.values.get("WaId", "")
    incoming_msg = request.values.get("Body", "").lower()
    print("Question: ", incoming_msg)
    add_message_to_history(wa_id, "Customer", incoming_msg)
    # Generate the answer using GPT-4
    answer = generate_answer(incoming_msg, wa_id)
    print("BOT Answer: ", answer)
    add_message_to_history(wa_id, "Bot", answer)
    bot_resp = MessagingResponse()
    bot_resp.message(answer)
    return str(bot_resp)
    # except Exception as e:
        # print("Error: ", e)
        # return str(MessagingResponse().message("An error occurred."))

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=8000)
