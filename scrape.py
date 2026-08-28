import requests
from bs4 import BeautifulSoup
import os
import json
import time
from urllib.parse import urljoin, urlparse
from collections import deque

BASE_URL = "https://www.uetmardan.edu.pk"
START_PATH = "/uetm/"

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_PAGES = 500       # safety cap so it doesn't run forever
DELAY_SECONDS = 1      # be polite to the server

# Skip links that are clearly not content pages (login, downloads handled separately, etc.)
SKIP_KEYWORDS = [
    "javascript:", "mailto:", ".pdf", ".jpg", ".jpeg", ".png", ".gif",
    "logout", "login", "#"
]

visited = set()
to_visit = deque([START_PATH])
manifest = []


def is_internal(url):
    parsed = urlparse(url)
    return parsed.netloc == "" or "uetmardan.edu.pk" in parsed.netloc


def should_skip(url):
    return any(kw in url.lower() for kw in SKIP_KEYWORDS)


def clean_filename(url_path):
    name = url_path.strip("/").replace("/", "_")
    return name if name else "home"


def scrape_page(url_path):
    full_url = urljoin(BASE_URL, url_path)
    try:
        response = requests.get(full_url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"FAILED: {full_url} -> {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    # Collect internal links before stripping anything
    links_found = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if should_skip(href):
            continue
        joined = urljoin(full_url, href)
        if is_internal(joined):
            links_found.append(joined)

    # Remove noise elements for text extraction
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else url_path
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    if len(clean_text) >= 50:
        filename = clean_filename(urlparse(full_url).path) + ".txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(clean_text)
        manifest.append({"filename": filename, "url": full_url, "title": title})
        print(f"SAVED ({len(visited)+1}): {full_url}")
    else:
        print(f"SKIPPED (too little content): {full_url}")

    return links_found


if __name__ == "__main__":
    while to_visit and len(visited) < MAX_PAGES:
        path = to_visit.popleft()
        full_url = urljoin(BASE_URL, path)
        norm_url = full_url.split("#")[0].rstrip("/")

        if norm_url in visited:
            continue
        visited.add(norm_url)

        found_links = scrape_page(path)
        time.sleep(DELAY_SECONDS)

        for link in found_links:
            norm_link = link.split("#")[0].rstrip("/")
            if norm_link not in visited:
                to_visit.append(link)

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Scraped {len(manifest)} pages out of {len(visited)} visited. See {OUTPUT_DIR}/manifest.json")