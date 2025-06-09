import requests
from bs4 import BeautifulSoup
import os
import re

urls = [
    "https://www.britannica.com/place/France/Land",
    "https://www.britannica.com/place/France/The-Hercynian-massifs",
    "https://www.britannica.com/place/France/The-great-lowlands",
    "https://www.britannica.com/place/France/The-younger-mountains-and-adjacent-plains",
    "https://www.britannica.com/place/France/Drainage",
    "https://www.britannica.com/place/France/Soils",
    "https://www.britannica.com/place/France/Climate",
    "https://www.britannica.com/place/France/Plant-and-animal-life"
]

headers = {"User-Agent": "Mozilla/5.0 (compatible; my-scraper/1.0)"}

# Create an output directory
output_dir = "data"
# os.makedirs(output_dir, exist_ok=True)

def sanitize_filename(title):
    return re.sub(r'[\\/*?:"<>|]', "", title)

for url in urls:
    print(f"Fetching {url} …")
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.content, "html.parser")

    # Get a nice title for the file
    title_tag = soup.find("title")
    title = title_tag.text.split("|")[0].strip() if title_tag else "Untitled"
    filename = sanitize_filename(title) + ".txt"

    # Extract main content
    sections = soup.find_all("section")
    page_text = ""

    for sec in sections:
        headings = sec.find_all(["h1", "h2", "h3", "h4"])
        for h in headings:
            page_text += f"\n\n{h.get_text(strip=True).upper()}\n"

        text = sec.get_text(separator=" ", strip=True)
        page_text += text + "\n\n"

    if not page_text.strip():
        # Fallback if section parsing fails
        paragraphs = soup.find_all("p")
        page_text = "\n\n".join(p.get_text(strip=True) for p in paragraphs)

    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        # f.write(f"URL: {url}\n\n")
        f.write(page_text.strip())

    print(f"Saved: {filename}")

print("✅ Done! All pages saved in 'britannica_france_pages/' folder.")
