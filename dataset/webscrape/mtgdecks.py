import requests
import os
from tqdm import tqdm
from bs4 import BeautifulSoup
import re

def main():
    # Initialize session and response
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    url = "https://mtgdecks.net"
    download_folder = "./data/mtg_decks"

    # Make sure the download folder exists
    os.makedirs(download_folder, exist_ok=True)

    deck_per_page = 15
    decks_downloaded = len(os.listdir(download_folder))
    page_num = decks_downloaded // deck_per_page
    for _ in tqdm(range(page_num, 865)):
        deck_url = f"{url}/Standard/decklists/page:{page_num}"
        response = session.get(deck_url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        if response.status_code != 200:
            break

        # Find the table containing the deck data
        table = soup.find('table', class_='clickable table table-striped hidden-xs')

        # If no table is found, assume there are no more pages and break the loop
        if not table:
            break

        # Iterate over each row in the table, skipping the header
        for row in table.find_all('tr')[1:]:  # Assuming the first row is the header
            columns = row.find_all('td')
            if len(columns) > 2:  # Ensure there are enough columns to prevent index errors
                deck_link = columns[2].find('a')
                deck_name = deck_link.text.strip().lower().replace(' ', '-').replace('.', '').replace('"', '')
                player_info = columns[2].find('strong')
                player_name = player_info.text.strip('By ').strip().lower().replace(' ', '-').replace('.', '').replace('"', '')

                link = deck_link['href']
                download_link = f'{url}{link}/txt'
                file_name = f"Standard_{deck_name}_by_{player_name}.txt".replace("/", "").replace("\\", "").replace("?", "").replace('|', '')
                file_path = os.path.join(download_folder, file_name)

                # Download the deck file
                deck_response = session.get(download_link, headers=headers)
                if deck_response.status_code == 200:
                    try:
                        with open(file_path, 'wb') as file:
                            file.write(deck_response.content)
                        # print(f"Downloaded: {file_name}")
                    except:
                        print(f"Failed to download: {file_name}")
                else:
                    print(f"Failed to download: {file_name}")


        page_num += 1

if __name__ == "__main__":
    main()