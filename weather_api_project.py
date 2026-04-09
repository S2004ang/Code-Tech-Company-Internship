import requests

API_KEY = "37fc9d3f7e8180c017182705170490e6"
CITY = "Hyderabad"

url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

response = requests.get(url)
data = response.json()

print(data)   # ADD THIS LINE

temperature = data["main"]["temp"]
humidity = data["main"]["humidity"]
wind_speed = data["wind"]["speed"]

print("City:", CITY)
print("Temperature:", temperature)
print("Humidity:", humidity)
print("Wind Speed:", wind_speed)



import pandas as pd

weather_data = {
    "Parameter": ["Temperature", "Humidity", "Wind Speed"],
    "Value": [temperature, humidity, wind_speed]
}

df = pd.DataFrame(weather_data)

print(df)

import matplotlib.pyplot as plt

parameters = ["Temperature", "Humidity", "Wind Speed"]
values = [temperature, humidity, wind_speed]

plt.bar(parameters, values)

plt.title("Weather Data Visualization")
plt.xlabel("Weather Parameters")
plt.ylabel("Values")

plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(x=parameters, y=values)

plt.title("Weather Data Visualization")
plt.show()

plt.savefig("weather_chart.png")