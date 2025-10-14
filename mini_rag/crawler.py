import os
import time
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin


def crawl_site(start_url: str, max_pages: int = 10, max_depth: int = 2, crawl_delay_ms: int = 500):
    """
    Crawl in-domain pages starting from a URL, respecting domain scope.
    Saves extracted text content into artifacts/pages/combined_text.json
    """

    visited = set()
    pages = []
    domain = urlparse(start_url).netloc
    queue = [(start_url, 0)]

    print(f"[INFO] Starting crawl from {start_url} (max_pages={max_pages}, max_depth={max_depth})")

    while queue and len(pages) < max_pages:
        url, depth = queue.pop(0)
        if url in visited or depth > max_depth:
            continue

        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "mini-rag-crawler"})
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            for s in soup(["script", "style", "noscript"]):
                s.extract()
            text = " ".join(soup.stripped_strings)

            if text:
                pages.append({"url": url, "text": text})
                print(f"[OK] Crawled: {url} ({len(text)} chars)")

            for link_tag in soup.find_all("a", href=True):
                next_url = urljoin(url, link_tag["href"])
                next_domain = urlparse(next_url).netloc
                if next_domain == domain and next_url not in visited:
                    queue.append((next_url, depth + 1))

            visited.add(url)
            time.sleep(crawl_delay_ms / 1000.0)

        except Exception as e:
            print(f"[WARN] Skipping {url}: {e}")
            continue

    os.makedirs("artifacts/pages", exist_ok=True)

    with open("artifacts/pages/combined_text.json", "w") as f:
        json.dump(pages, f, indent=2)

    urls = [p["url"] for p in pages]
    print(f"[DONE] Crawled {len(pages)} pages, saved to artifacts/pages/combined_text.json")

    return pages, 0, urls