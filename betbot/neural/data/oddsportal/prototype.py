import requests
from bs4 import BeautifulSoup


def load_odds(country: str, league: str, season: int):
    url = f"https://www.oddsportal.com/soccer/" \
          f"{country}/{league}-{season}-{season + 1}/results/"
    print(url)
    data = requests.get(url)
    soup = BeautifulSoup(data.text, "html.parser")

    for x in soup.select(".deactivate"):
        print(x.text)


if __name__ == '__main__':
    load_odds("germany", "bundesliga", 2019)
