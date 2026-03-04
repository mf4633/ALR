import requests

lat, lon = 38.4351, -78.8740  # Harrisonburg, VA
url = f"https://hdsc.nws.noaa.gov/cgi-bin/new/cgi_readH5.py?lat={lat:.4f}&lon={lon:.4f}&type=pf"
headers = {'User-Agent': 'PCSWMM-Rain-Grid-Creator/1.0'}
response = requests.get(url, headers=headers, timeout=30)
print("Status Code:", response.status_code)
print("Response Text:", response.text)