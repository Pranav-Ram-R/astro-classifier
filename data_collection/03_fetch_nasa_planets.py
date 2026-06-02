"""
03_fetch_nasa_planets.py  (v3 — Solar System objects)
------------------------------------------------------
Fetch images of Solar System objects (planets, moons, dwarf planets)
from the NASA Image and Video Library API.

Changes from v2:
  • Class is now "Solar System object", not just planet. Major moons and
    dwarf planets are included alongside the 8 planets.
  • Per-target cap (MAX_PER_TARGET) prevents the dataset from being
    dominated by a single body (e.g. the Moon, which has thousands of
    Apollo photos).
  • Slightly expanded noise filter for moon-specific junk
    (astronaut close-ups, sample bags, spacesuits, footprints).

API docs: https://images.nasa.gov/docs/images.nasa.gov_api_docs.pdf

Run:
    python 03_fetch_nasa_planets.py

Output:
    data/processed/planetary/{nasa_id}.jpg
"""

from __future__ import annotations

import random
import re
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("data/processed/planetary")

API_SEARCH = "https://images-api.nasa.gov/search"
API_ASSET  = "https://images-api.nasa.gov/asset"

# Queries grouped by target body. Each target's queries run until the
# per-target cap is hit or the queries are exhausted.
TARGET_QUERIES: dict[str, list[str]] = {
    # --- Planets ---
    "Jupiter":   ["Jupiter planet", "Jupiter Juno photograph",
                  "Jupiter Voyager", "Jupiter Cassini",
                  "Jupiter Great Red Spot"],
    "Saturn":    ["Saturn planet", "Saturn Cassini",
                  "Saturn rings photograph", "Saturn Voyager"],
    "Mars":      ["Mars planet global view", "Mars surface Curiosity",
                  "Mars Perseverance photograph",
                  "Mars Reconnaissance Orbiter", "Mars Olympus Mons"],
    "Neptune":   ["Neptune planet Voyager", "Neptune planet"],
    "Uranus":    ["Uranus planet Voyager", "Uranus planet"],
    "Venus":     ["Venus planet Magellan", "Venus Mariner",
                  "Venus surface"],
    "Mercury":   ["Mercury planet MESSENGER", "Mercury planet Mariner"],
    "Earth":     ["Earth from space full disk", "Earth Blue Marble",
                  "Earth Apollo photograph"],
    "Pluto":     ["Pluto New Horizons"],

    # --- Major moons ---
    "Moon":      ["Moon LRO photograph", "Moon full disk",
                  "Moon Apollo photograph from orbit",
                  "lunar surface photograph"],
    "Io":        ["Io moon Jupiter", "Io volcano Galileo spacecraft"],
    "Europa":    ["Europa moon Jupiter", "Europa Galileo spacecraft"],
    "Ganymede":  ["Ganymede moon Jupiter"],
    "Callisto":  ["Callisto moon Jupiter"],
    "Titan":     ["Titan moon Saturn Cassini", "Titan haze atmosphere"],
    "Enceladus": ["Enceladus moon Saturn", "Enceladus plumes Cassini"],
    "Mimas":     ["Mimas moon Saturn"],
    "Iapetus":   ["Iapetus moon Saturn"],
    "Rhea":      ["Rhea moon Saturn"],
    "Triton":    ["Triton moon Neptune Voyager"],
    "Charon":    ["Charon moon Pluto New Horizons"],

    # --- Dwarf planets ---
    "Ceres":     ["Ceres dwarf planet Dawn"],
}

MAX_PER_TARGET = 60        # cap so no single body dominates
MAX_PAGES_PER_QUERY = 5    # 100 results per page; usually hit cap first

# ---- Noise filter ----
NOISE_WORDS = [
    "artist", "illustration", "concept", "diagram", "infographic",
    "logo", "patch", "insignia", "badge", "emblem",
    "poster", "stamp", "cartoon", "comic", "clipart",
    "team photo", "group photo", "portrait", "selfie",
    "press conference", "press release", "ceremony",
    "blueprint", "schematic", "cutaway",
    "montage", "collage", "banner", "brochure",
    "animation", "simulated", "rendering",
    # Moon/Mars surface noise (people, hardware, samples):
    "astronaut", "spacesuit", "spacewalk",
    "lunar sample", "rock sample", "soil sample",
    "footprint",
]
_NOISE_RE = re.compile("|".join(re.escape(w) for w in NOISE_WORDS),
                        re.IGNORECASE)

MIN_IMAGE_BYTES = 30_000

REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
RATE_LIMIT_API = 1.0
RATE_LIMIT_DOWNLOAD = 1.5
RATE_LIMIT_JITTER = 0.3

USER_AGENT = (
    "AstroClassifierFetcher/3.0 "
    "(educational project; respectful of rate limits)"
)


# ---------------------------------------------------------------------------
# HTTP plumbing
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


def polite_sleep(base: float) -> None:
    time.sleep(max(0.3, base + random.uniform(-RATE_LIMIT_JITTER,
                                               RATE_LIMIT_JITTER)))


def fetch_json(session: requests.Session, url: str,
               params: dict | None = None) -> dict | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"      429 rate-limited; waiting {wait:.0f}s",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                ConnectionResetError) as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"      [retry {attempt}] {type(e).__name__}; "
                  f"waiting {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def search_page(session: requests.Session, query: str,
                page: int) -> list[dict]:
    data = fetch_json(session, API_SEARCH, params={
        "q": query, "media_type": "image", "page": page,
    })
    if not data:
        return []
    return data.get("collection", {}).get("items", [])


def get_original_jpg_url(session: requests.Session,
                         nasa_id: str) -> str | None:
    data = fetch_json(session, f"{API_ASSET}/{nasa_id}")
    if not data:
        return None
    items = data.get("collection", {}).get("items", [])
    best: str | None = None
    best_rank = 99
    for item in items:
        href: str = item.get("href", "")
        lower = href.lower()
        if not lower.endswith((".jpg", ".jpeg")):
            continue
        if "~orig" in lower:    rank = 0
        elif "~large" in lower: rank = 1
        elif "~medium" in lower: rank = 2
        else: rank = 3
        if rank < best_rank:
            best_rank = rank
            best = href
    return best


def is_noise(title: str) -> bool:
    return bool(_NOISE_RE.search(title))


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_image(session: requests.Session, url: str,
                   out_path: Path) -> int:
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    written = 0
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(64 * 1024):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    if written < MIN_IMAGE_BYTES:
        tmp.unlink(missing_ok=True)
        return 0

    tmp.replace(out_path)
    return written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def collect_candidates(session: requests.Session) -> list[tuple[str, str, str]]:
    """Returns list of (target, nasa_id, title). Enforces MAX_PER_TARGET."""
    seen_ids: set[str] = set()
    candidates: list[tuple[str, str, str]] = []
    per_target_counts: dict[str, int] = {t: 0 for t in TARGET_QUERIES}

    total_queries = sum(len(qs) for qs in TARGET_QUERIES.values())
    qi = 0

    for target, queries in TARGET_QUERIES.items():
        for query in queries:
            qi += 1
            if per_target_counts[target] >= MAX_PER_TARGET:
                break

            for page in range(1, MAX_PAGES_PER_QUERY + 1):
                if per_target_counts[target] >= MAX_PER_TARGET:
                    break

                print(f"  [{qi}/{total_queries}] target={target:<10} "
                      f"q=\"{query[:35]:<35}\" page {page}  "
                      f"({len(candidates)} total, "
                      f"{per_target_counts[target]}/{MAX_PER_TARGET} for {target})",
                      end="\r", flush=True)

                items = search_page(session, query, page)
                polite_sleep(RATE_LIMIT_API)
                if not items:
                    break

                new_on_page = 0
                for item in items:
                    if per_target_counts[target] >= MAX_PER_TARGET:
                        break
                    meta_list = item.get("data", [])
                    if not meta_list:
                        continue
                    meta = meta_list[0]
                    nasa_id = meta.get("nasa_id", "")
                    title = meta.get("title", "")
                    if not nasa_id or nasa_id in seen_ids:
                        continue
                    if is_noise(title):
                        continue
                    seen_ids.add(nasa_id)
                    candidates.append((target, nasa_id, title))
                    per_target_counts[target] += 1
                    new_on_page += 1

                if new_on_page == 0:
                    break

    print()
    print("\n  Per-target candidate counts:")
    for target, count in per_target_counts.items():
        bar = "█" * (count // 3)
        print(f"    {target:<10} {count:>3}  {bar}")
    return candidates


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    print("=" * 60)
    print("NASA Image & Video Library — Solar System object scraper v3")
    print("=" * 60)

    print("\n[1/2] Collecting candidate image IDs from API search\n")
    candidates = collect_candidates(session)
    print(f"\n  Total candidates after de-dup + noise filter + cap: "
          f"{len(candidates)}\n")

    print("[2/2] Downloading images\n")
    stats = {"downloaded": 0, "skipped": 0, "too_small": 0, "failed": 0}

    try:
        for idx, (target, nasa_id, title) in enumerate(candidates, 1):
            safe_id = re.sub(r'[^\w\-]', '_', nasa_id)
            out_path = OUTPUT_DIR / f"{safe_id}.jpg"

            if out_path.exists() and out_path.stat().st_size > MIN_IMAGE_BYTES:
                stats["skipped"] += 1
                if idx % 50 == 0:
                    print(f"  [{idx:>5}/{len(candidates)}] ... "
                          f"({stats['skipped']} skipped so far)")
                continue

            print(f"  [{idx:>5}/{len(candidates)}] {target:<10} "
                  f"{safe_id:<28} ", end="", flush=True)

            try:
                jpg_url = get_original_jpg_url(session, nasa_id)
                polite_sleep(RATE_LIMIT_API)
                if not jpg_url:
                    print("FAIL (no JPEG in asset manifest)")
                    stats["failed"] += 1
                    continue

                n_bytes = download_image(session, jpg_url, out_path)
                polite_sleep(RATE_LIMIT_DOWNLOAD)

                if n_bytes == 0:
                    print("SKIP (too small)")
                    stats["too_small"] += 1
                    continue

                print(f"OK ({n_bytes // 1024} KB)")
                stats["downloaded"] += 1

            except KeyboardInterrupt:
                print(" INTERRUPTED")
                raise
            except Exception as e:
                print(f"FAIL ({type(e).__name__}: {e})")
                stats["failed"] += 1
                out_path.with_suffix(out_path.suffix + ".part").unlink(
                    missing_ok=True)
                continue
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run to resume.")
        return 130

    print("\n" + "=" * 60)
    print("OVERALL")
    print("=" * 60)
    print(f"  Candidates:   {len(candidates)}")
    print(f"  Downloaded:   {stats['downloaded']}")
    print(f"  Skipped:      {stats['skipped']} (already on disk)")
    print(f"  Too small:    {stats['too_small']} (<{MIN_IMAGE_BYTES//1024} KB)")
    print(f"  Failed:       {stats['failed']}")
    print(f"  Output dir:   {OUTPUT_DIR}")
    print("\n  Manual QA still recommended — skim the folder for obvious")
    print("  junk (instrument close-ups, calibration targets, etc.)")
    print("  and delete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())