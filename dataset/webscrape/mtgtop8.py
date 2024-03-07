import os
import re
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

# Base URL for form submission
base_url = "https://www.mtgtop8.com/"
url = base_url + "search"

# Specify the folder where you want to save the downloaded files
download_folder = "./data/mt_top8"

# Make sure the download folder exists
os.makedirs(download_folder, exist_ok=True)

# Assuming 'ST' represents Standard format in their system
data = {
    "format": "ST",
    # Your other parameters here
    "current_page": 1,  # Start with page 1
}

session = requests.Session()

# Get the total number of decks and calculate the total pages
response = session.post(url, data=data)
soup = BeautifulSoup(response.content, "html.parser")
total_decks = int(soup.find('div', class_='w_title').text.split()[0])
items_per_page = 25
total_pages = (total_decks + items_per_page - 1) // items_per_page

# Iterate over the pages
for current_page in tqdm(range(1, total_pages + 1), desc="Processing pages"):
    # Update the page number in the form data
    data["current_page"] = current_page
    
    response = session.post(url, data=data)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        rows = soup.find_all('tr', class_='hover_tr')
        
        for row in rows:
            deck_cell = row.find('td', class_='S12')
            deck_name = deck_cell.get_text(strip=True).replace(" ", "_")
            deck_link = deck_cell.find('a')['href']
            
            full_deck_link = f"{base_url}{deck_link}"
            player_name = row.find('td', class_='G12').get_text(strip=True).replace(" ", "_")
            
            f_string = full_deck_link[-11:-3]
            download_link = f"{base_url}mtgo?d={f_string}=Standard_{deck_name}_by_{player_name}"
            file_name = f"Standard_{deck_name}_by_{player_name}.txt".replace("/", "").replace("\"", "").replace("?", "")
            file_path = os.path.join(download_folder, file_name)
            
            if os.path.exists(file_path):
                continue
            
            response = requests.get(download_link)
            try:
                with open(file_path, 'wb') as file:
                    file.write(response.content)
            except:
                continue
    else:
        print("Failed to retrieve data")
        break