import requests
url = 'http://espheartbeat.local:80/upload'  # Update the URL accordingly
payload = {
       "hr_fourier_chrom": 100
}

try:
   response = requests.post(url, json=payload)
   if response.status_code == 200:
      print("Data successfully sent to the endpoint")
   else:
      print("Failed to send data to the endpoint. Status code:", response.status_code)
except Exception as e:
   print("An error occurred:", e)
   print("kokot")
