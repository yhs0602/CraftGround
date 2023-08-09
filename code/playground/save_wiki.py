import requests
from bs4 import BeautifulSoup

if __name__ == "__main__":
    url = "https://minecraft.fandom.com/wiki/Fishing"
    html = requests.get(url)
    soup = BeautifulSoup(html.text, "html.parser")

    content = soup.get_text("\n", strip=True)
    with open("fishing.txt", "w") as f:
        f.write(content)
