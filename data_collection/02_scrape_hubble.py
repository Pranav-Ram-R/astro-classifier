"""
02_scrape_hubble.py
-------------------
Scrape nebula and star-cluster JPEGs from ESA/Hubble for the astronomical
object classifier dataset.

The previous HTML-scraping approach broke because the category pages no
longer expose image links in <a href> tags (the IDs are present as plain
text in the HTML but not in any clickable anchor). This script instead
uses the site's RSS feeds, which still expose clean <item><link> entries
to image detail pages. The detail pages themselves still have normal
anchor tags to the JPEG downloads.

Run:
    python 02_scrape_hubble.py

Outputs:
    data/processed/nebula/{image_id}.jpg
    data/processed/star_cluster/{image_id}.jpg

Resumable: existing files are skipped. Safe to interrupt and re-run.
"""

from __future__ import annotations

import random
import re
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RSS_FEEDS: dict[str, str] = {
    "nebula":       "https://esahubble.org/images/feed/category/nebulae/",
    "star_cluster": "https://esahubble.org/images/feed/category/starclusters/",
}

OUTPUT_BASE = Path("data/processed")

REQUEST_TIMEOUT = 30          # seconds per HTTP request
MAX_RETRIES = 5               # outer retry loop for connection-reset etc.
RATE_LIMIT_SECONDS = 1.5      # base delay between requests
RATE_LIMIT_JITTER = 0.4       # +/- this much randomness on each delay

USER_AGENT = (
    "AstroClassifierScraper/1.0 "
    "(educational project; respectful of robots.txt and rate limits)"
)

# Detail pages offer several JPEG sizes. Try in this order of preference.
# `publicationjpg` is high-quality but typically only a few MB; `large` and
# `wallpaper1` are good fallbacks; `screen` is the smallest usable.
JPEG_PREFERENCE = ["publicationjpg", "large", "wallpaper1", "screen"]


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    """A Session with sensible retry/backoff for transient HTTP failures."""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=1.5,                          # 1.5, 3, 6, 12, 24 s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def polite_sleep() -> None:
    delay = RATE_LIMIT_SECONDS + random.uniform(-RATE_LIMIT_JITTER,
                                                 RATE_LIMIT_JITTER)
    time.sleep(max(0.5, delay))


def fetch_with_retries(session: requests.Session,
                       url: str,
                       *,
                       stream: bool = False,
                       attempts: int = MAX_RETRIES) -> requests.Response:
    """
    Outer retry layer on top of urllib3's Retry.

    urllib3's Retry handles HTTP status codes and many connection errors,
    but on Windows we still occasionally see ConnectionResetError (WinError
    10054) escape as a bare exception. Wrapping the call here makes the
    script robust to that.
    """
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT, stream=stream)
            response.raise_for_status()
            return response
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ReadTimeout,
                ConnectionResetError) as e:
            last_err = e
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"      [retry {attempt}/{attempts}] "
                  f"{type(e).__name__}: {e}; waiting {wait:.1f}s",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {attempts} attempts: {url}") from last_err


# ---------------------------------------------------------------------------
# RSS parsing
# ---------------------------------------------------------------------------

_IMAGE_ID_RE = re.compile(r"/images/([A-Za-z0-9_-]+)/?$")


def parse_rss_items(rss_xml: str) -> list[tuple[str, str]]:
    """
    Return a list of (image_id, detail_url) tuples from an ESA/Hubble RSS feed.

    Each <item><link> looks like https://esahubble.org/images/heic1234a/ ;
    the image ID is the trailing path component.
    """
    root = ET.fromstring(rss_xml)
    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in root.iter("item"):
        link_el = item.find("link")
        if link_el is None or not link_el.text:
            continue
        url = link_el.text.strip()
        m = _IMAGE_ID_RE.search(url)
        if not m:
            continue
        image_id = m.group(1)
        if image_id in seen:
            continue
        seen.add(image_id)
        items.append((image_id, url))
    return items


# ---------------------------------------------------------------------------
# Detail-page parsing
# ---------------------------------------------------------------------------

def find_jpeg_url(detail_html: str, image_id: str) -> str | None:
    """
    Scan an image detail page for the best JPEG download link.

    The page typically has several download options for different sizes.
    We pick by JPEG_PREFERENCE order; if none of the preferred sizes are
    found, we return the first .jpg link that references this image_id.
    """
    soup = BeautifulSoup(detail_html, "html.parser")
    candidates: list[tuple[int, str]] = []  # (rank, url)

    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if not href.lower().endswith(".jpg"):
            continue
        if image_id not in href:
            continue

        # Normalise to absolute URL
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = "https://esahubble.org" + href

        # Rank by which size folder it lives in
        rank = len(JPEG_PREFERENCE)  # default = lowest priority
        for i, pref in enumerate(JPEG_PREFERENCE):
            if f"/{pref}/" in href:
                rank = i
                break
        candidates.append((rank, href))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_jpeg(session: requests.Session,
                  url: str,
                  out_path: Path) -> int:
    """
    Stream a JPEG to disk via a .part temp file, then atomic-rename.

    Returns the number of bytes written.
    """
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    bytes_written = 0
    response = fetch_with_retries(session, url, stream=True)
    try:
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)
    finally:
        response.close()

    if bytes_written == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded 0 bytes")

    tmp_path.replace(out_path)
    return bytes_written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def scrape_category(session: requests.Session,
                    category: str,
                    feed_url: str,
                    out_dir: Path) -> dict[str, int]:
    print(f"\n=== {category} ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  RSS feed: {feed_url}")
    try:
        rss_response = fetch_with_retries(session, feed_url)
    except Exception as e:
        print(f"  ERROR fetching RSS feed: {e}", file=sys.stderr)
        return {"downloaded": 0, "skipped": 0, "failed": 0, "total": 0}
    polite_sleep()

    try:
        items = parse_rss_items(rss_response.text)
    except ET.ParseError as e:
        print(f"  ERROR parsing RSS XML: {e}", file=sys.stderr)
        return {"downloaded": 0, "skipped": 0, "failed": 0, "total": 0}

    print(f"  RSS returned {len(items)} unique image IDs")

    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "total": len(items)}

    for idx, (image_id, detail_url) in enumerate(items, 1):
        out_path = out_dir / f"{image_id}.jpg"

        if out_path.exists() and out_path.stat().st_size > 0:
            stats["skipped"] += 1
            print(f"  [{idx:>4}/{len(items)}] {image_id:<20} SKIP (exists)")
            continue

        print(f"  [{idx:>4}/{len(items)}] {image_id:<20} ", end="", flush=True)
        try:
            detail_response = fetch_with_retries(session, detail_url)
            polite_sleep()

            jpeg_url = find_jpeg_url(detail_response.text, image_id)
            if not jpeg_url:
                print("FAIL (no JPEG link found on detail page)")
                stats["failed"] += 1
                continue

            n_bytes = download_jpeg(session, jpeg_url, out_path)
            print(f"OK ({n_bytes // 1024} KB)")
            stats["downloaded"] += 1
            polite_sleep()

        except KeyboardInterrupt:
            print(" INTERRUPTED")
            raise
        except Exception as e:
            print(f"FAIL ({type(e).__name__}: {e})")
            stats["failed"] += 1
            # Clean up any leftover .part file
            (out_path.with_suffix(out_path.suffix + ".part")).unlink(
                missing_ok=True
            )
            continue

    print(f"\n  Summary for {category}:")
    print(f"    Downloaded: {stats['downloaded']}")
    print(f"    Skipped:    {stats['skipped']}")
    print(f"    Failed:     {stats['failed']}")
    print(f"    Output dir: {out_dir}")
    return stats


def main() -> int:
    session = make_session()
    grand_total = {"downloaded": 0, "skipped": 0, "failed": 0, "total": 0}

    try:
        for category, feed_url in RSS_FEEDS.items():
            out_dir = OUTPUT_BASE / category
            stats = scrape_category(session, category, feed_url, out_dir)
            for k in grand_total:
                grand_total[k] += stats[k]
    except KeyboardInterrupt:
        print("\nInterrupted by user. Re-run to resume; existing files are kept.")
        return 130

    print("\n" + "=" * 60)
    print("OVERALL")
    print("=" * 60)
    print(f"  Total IDs seen: {grand_total['total']}")
    print(f"  Downloaded:     {grand_total['downloaded']}")
    print(f"  Skipped:        {grand_total['skipped']}")
    print(f"  Failed:         {grand_total['failed']}")
    print("\nRe-run anytime to resume. Existing files are skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())