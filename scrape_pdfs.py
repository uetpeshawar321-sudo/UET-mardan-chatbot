import requests
from bs4 import BeautifulSoup
import os
import json
import time
from urllib.parse import urljoin, urlparse
from collections import deque
from pypdf import PdfReader

BASE_URL = "https://www.uetmardan.edu.pk"
START_PATH = "/uetm/"

PDF_DIR = "data/pdfs"
TEXT_DIR = "data/raw"  # extracted PDF text goes alongside the scraped page text
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

MAX_PAGES = 500
DELAY_SECONDS = 1

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

SKIP_KEYWORDS = ["javascript:", "mailto:", ".jpg", ".jpeg", ".png", ".gif", "logout", "login", "#"]

visited = set()
to_visit = deque([START_PATH])
pdf_manifest = []
found_pdf_urls = set()


def normalize_url(url):
    parsed = urlparse(url)
    netloc = parsed.netloc.replace("www.", "")
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{netloc}{path}"


def is_internal(url):
    parsed = urlparse(url)
    return parsed.netloc == "" or "uetmardan.edu.pk" in parsed.netloc


def should_skip(url):
    return any(kw in url.lower() for kw in SKIP_KEYWORDS)


def clean_filename(url_path):
    name = url_path.strip("/").replace("/", "_")
    return name if name else "home"


def download_and_extract_pdf(pdf_url):
    """Download a PDF, extract its text, save both the PDF and the extracted text."""
    try:
        response = requests.get(pdf_url, timeout=30, headers=HEADERS)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"PDF FAILED (download): {pdf_url} -> {e}")
        return

    filename_base = clean_filename(urlparse(pdf_url).path)
    pdf_path = os.path.join(PDF_DIR, filename_base + ".pdf")

    with open(pdf_path, "wb") as f:
        f.write(response.content)

    # Extract text
    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        full_text = "\n".join(text_parts).strip()
    except Exception as e:
        print(f"PDF FAILED (extract text): {pdf_url} -> {e}")
        return

    if len(full_text) < 30:
        print(f"PDF SKIPPED (no extractable text, likely scanned image): {pdf_url}")
        return

    txt_filename = filename_base + "_pdf.txt"
    txt_path = os.path.join(TEXT_DIR, txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    pdf_manifest.append({
        "filename": txt_filename,
        "source_pdf": pdf_path,
        "url": pdf_url
    })
    print(f"PDF SAVED: {pdf_url} -> {txt_filename}")


def crawl_page_for_links(url_path):
    full_url = urljoin(BASE_URL, url_path)
    try:
        response = requests.get(full_url, timeout=20, headers=HEADERS)
        response.raise_for_status()
    except requests.RequestException:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    base_tag = soup.find("base", href=True)
    resolve_base = urljoin(full_url, base_tag["href"]) if base_tag else full_url

    page_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if should_skip(href) and ".pdf" not in href.lower():
            continue
        joined = urljoin(resolve_base, href)
        if not is_internal(joined):
            continue

        if joined.lower().endswith(".pdf"):
            found_pdf_urls.add(joined)
        else:
            page_links.append(joined)

    return page_links


if __name__ == "__main__":
    print("Crawling site to discover PDF links...")
    while to_visit and len(visited) < MAX_PAGES:
        path = to_visit.popleft()
        full_url = urljoin(BASE_URL, path)
        norm_url = normalize_url(full_url)

        if norm_url in visited:
            continue
        visited.add(norm_url)

        links = crawl_page_for_links(path)
        time.sleep(DELAY_SECONDS)

        for link in links:
            norm_link = normalize_url(link)
            if norm_link not in visited:
                to_visit.append(link)

    print(f"\nFound {len(found_pdf_urls)} unique PDF links. Downloading and extracting text...\n")

    for pdf_url in found_pdf_urls:
        download_and_extract_pdf(pdf_url)
        time.sleep(DELAY_SECONDS)

    with open(os.path.join(PDF_DIR, "pdf_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(pdf_manifest, f, indent=2)

    print(f"\nDone. Extracted text from {len(pdf_manifest)} PDFs out of {len(found_pdf_urls)} found.")
    print(f"See {PDF_DIR}/pdf_manifest.json")