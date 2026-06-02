"""
02b_scrape_hubble_starclusters.py
---------------------------------
Supplemental scraper for ESA/Hubble star cluster images.

Why a second script?
--------------------
02_scrape_hubble.py uses the RSS feed, which only exposes ~25 recent
items per category. The full category archive lists 300+ star clusters,
but their image IDs are only present as plain text in the HTML — they
aren't wrapped in <a href> tags, which is what broke the original
scraper. The fix is to fetch the paginated archive pages and extract
IDs with a regex over the raw HTML, then hand each ID off to the same
detail-page → JPEG-download logic used by the RSS scraper.

Run AFTER 02_scrape_hubble.py. Files already on disk are skipped, so the
25 RSS-sourced files won't be redownloaded.

Run:
    python 02b_scrape_hubble_starclusters.py

Output:
    data/processed/star_cluster/{image_id}.jpg
"""

from __future__ import annotations

import random
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CATEGORY_SLUG = "starclusters"
OUTPUT_DIR = Path("data/processed/star_cluster")

ARCHIVE_BASE = f"https://esahubble.org/images/archive/category/{CATEGORY_SLUG}/"
DETAIL_BASE = "https://esahubble.org/images/"

MAX_PAGES = 20                # safety cap on pagination
REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
RATE_LIMIT_SECONDS = 1.5
RATE_LIMIT_JITTER = 0.4

USER_AGENT = (
    "AstroClassifierScraper/1.0 "
    "(educational project; respectful of robots.txt and rate limits)"
)

JPEG_PREFERENCE = ["publicationjpg", "large", "wallpaper1", "screen"]

# Image-release prefixes used by ESA/Hubble. We intentionally exclude `ann`
# (announcements — not actual images). Format is prefix + 4 digits + optional
# single-letter suffix, e.g. heic2401a, potw2103a, opo0415a.
ID_REGEX = re.compile(
    r"\b((?:heic|potw|opo|opw|opp)\d{4}[a-z]?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# HTTP plumbing (duplicated from 02_scrape_hubble.py so this script is
# standalone — feel free to refactor into a shared module later).
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=1.5,
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
                       attempts: int = MAX_RETRIES,
                       allow_404: bool = False) -> requests.Response | None:
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT, stream=stream)
            if allow_404 and response.status_code == 404:
                return None
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
        except requests.exceptions.HTTPError:
            raise
    raise RuntimeError(f"Failed after {attempts} attempts: {url}") from last_err


# ---------------------------------------------------------------------------
# Listing-page scraping (the new bit)
# ---------------------------------------------------------------------------

def page_url(page_number: int) -> str:
    """Pagination URL. Page 1 has no `/page/N/` suffix."""
    if page_number == 1:
        return ARCHIVE_BASE
    return f"{ARCHIVE_BASE}page/{page_number}/"


def extract_image_ids(html: str) -> list[str]:
    """Pull all image IDs from raw HTML using the regex over plain text."""
    seen: set[str] = set()
    ordered: list[str] = []
    for match in ID_REGEX.finditer(html):
        image_id = match.group(1).lower()
        if image_id not in seen:
            seen.add(image_id)
            ordered.append(image_id)
    return ordered


def collect_all_ids(session: requests.Session) -> list[str]:
    """Paginate through the archive listing and return every unique image ID."""
    all_ids: list[str] = []
    seen: set[str] = set()
    empty_pages = 0

    for page in range(1, MAX_PAGES + 1):
        url = page_url(page)
        print(f"  Fetching listing page {page}: {url}")
        try:
            response = fetch_with_retries(session, url, allow_404=True)
        except Exception as e:
            print(f"    ERROR: {e}", file=sys.stderr)
            break
        if response is None:
            print("    404 — end of pagination")
            break

        ids = extract_image_ids(response.text)
        new_ids = [i for i in ids if i not in seen]
        for i in new_ids:
            seen.add(i)
            all_ids.append(i)

        print(f"    Found {len(ids)} IDs on page ({len(new_ids)} new, "
              f"{len(all_ids)} total)")

        polite_sleep()

        if not new_ids:
            empty_pages += 1
            if empty_pages >= 2:
                print("    No new IDs for 2 consecutive pages — stopping")
                break
        else:
            empty_pages = 0

    return all_ids


# ---------------------------------------------------------------------------
# Detail-page parsing + download (same approach as the RSS scraper)
# ---------------------------------------------------------------------------

def find_jpeg_url(detail_html: str, image_id: str) -> str | None:
    soup = BeautifulSoup(detail_html, "html.parser")
    candidates: list[tuple[int, str]] = []

    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if not href.lower().endswith(".jpg"):
            continue
        if image_id not in href.lower():
            continue
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = "https://esahubble.org" + href

        rank = len(JPEG_PREFERENCE)
        for i, pref in enumerate(JPEG_PREFERENCE):
            if f"/{pref}/" in href:
                rank = i
                break
        candidates.append((rank, href))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def download_jpeg(session: requests.Session,
                  url: str,
                  out_path: Path) -> int:
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

def main() -> int:
    session = make_session()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Scraping ESA/Hubble category: {CATEGORY_SLUG} ===")
    print("\n[1/2] Collecting image IDs from paginated listing")
    ids = collect_all_ids(session)
    print(f"\n  Collected {len(ids)} unique image IDs in total.")

    print(f"\n[2/2] Visiting detail pages and downloading JPEGs")
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    try:
        for idx, image_id in enumerate(ids, 1):
            out_path = OUTPUT_DIR / f"{image_id}.jpg"

            if out_path.exists() and out_path.stat().st_size > 0:
                stats["skipped"] += 1
                print(f"  [{idx:>4}/{len(ids)}] {image_id:<20} SKIP (exists)")
                continue

            print(f"  [{idx:>4}/{len(ids)}] {image_id:<20} ",
                  end="", flush=True)
            try:
                detail_url = f"{DETAIL_BASE}{image_id}/"
                detail_response = fetch_with_retries(
                    session, detail_url, allow_404=True
                )
                polite_sleep()

                if detail_response is None:
                    print("FAIL (detail page 404)")
                    stats["failed"] += 1
                    continue

                jpeg_url = find_jpeg_url(detail_response.text, image_id)
                if not jpeg_url:
                    print("FAIL (no JPEG link on detail page)")
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
                (out_path.with_suffix(out_path.suffix + ".part")).unlink(
                    missing_ok=True
                )
                continue
    except KeyboardInterrupt:
        print("\nInterrupted by user. Re-run to resume.")
        return 130

    print("\n" + "=" * 60)
    print(f"OVERALL ({CATEGORY_SLUG})")
    print("=" * 60)
    print(f"  IDs found:  {len(ids)}")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  Skipped:    {stats['skipped']}")
    print(f"  Failed:     {stats['failed']}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("\nRe-run anytime to resume. Existing files are kept.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
