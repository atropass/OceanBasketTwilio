import json
str = '''{"type": "options", "content": "здравствуйте", "summary": "The customer has greeted and is likely to proceed with a request or query.", "status": "processing", "message_to": "Здравствуйте! Спасибо, что обратились в Ocean Basket. Как я могу помочь вам сегодня?"}'''
print(json.loads(str))