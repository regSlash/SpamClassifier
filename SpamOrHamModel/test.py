import requests

url = 'http://localhost:8000/classify/'
data = {'email': 'Тут ваше електронне повідомлення'}

response = requests.post(url, data=data)

if response.status_code == 200:
    result = response.json()['result']
    print(result)
else:
    print('Помилка при виконанні POST-запиту')