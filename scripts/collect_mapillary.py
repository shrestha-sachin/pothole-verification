"""Download Mapillary photos near a known image or inside a small bounding box.

Example:
    python scripts/collect_mapillary.py --image-id 299972942579578 --sequence --limit 20 --radius-m 250 --output data/raw/mapillary/segment_001
"""

import argparse
import csv
import math
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE = "https://graph.mapillary.com/"
FIELDS = (
    "id,thumb_2048_url,thumb_1024_url,computed_geometry,"
    "captured_at,sequence,creator,compass_angle"
)
CSV_FIELDS = [
    "image_id", "filename", "longitude", "latitude", "captured_at",
    "sequence_id", "creator", "compass_angle", "distance_from_seed_m",
    "source_url", "license",
]


def make_session(token):
    session = requests.Session()
    session.headers.update({"Authorization": f"OAuth {token}"})
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def api_get(session, endpoint, params=None):
    url = urljoin(API_BASE, endpoint)
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "graph.mapillary.com":
        raise ValueError("Refusing a pagination URL outside the Mapillary API")
    response = session.get(url, params=params, timeout=60)
    if not response.ok:
        # Do not include the authenticated URL or access token in exceptions.
        raise RuntimeError(f"Mapillary API error {response.status_code}: {response.text[:300]}")
    return response.json()


def get_collection(session, endpoint, params, max_pages=30):
    """Follow supported Graph API paging links (if present)."""
    results = []
    next_url, next_params = endpoint, params
    visited = set()
    for _ in range(max_pages):
        request_key = (next_url, str(next_params))
        if request_key in visited:
            break
        visited.add(request_key)
        payload = api_get(session, next_url, next_params)
        results.extend(payload.get("data", []))
        next_url = (payload.get("paging") or {}).get("next")
        next_params = None  # Next page URL already contains its parameters.
        if not next_url:
            break
    else:
        print(f"Warning: stopped at {max_pages} API pages; results may be incomplete.")
    return results


def get_image(session, image_id):
    return api_get(session, str(image_id), {"fields": FIELDS})


def coords(data):
    values = (data.get("computed_geometry") or {}).get("coordinates")
    if not isinstance(values, list) or len(values) < 2:
        return None
    return float(values[0]), float(values[1])  # longitude, latitude


def distance_m(a, b):
    lon1, lat1 = a
    lon2, lat2 = b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371000 * 2 * math.asin(min(1.0, math.sqrt(h)))


def nearby_sequence_images(session, image_id, limit, radius_m, max_scan, max_pages):
    seed = get_image(session, image_id)
    origin = coords(seed)
    if origin is None:
        raise RuntimeError("Seed image is missing coordinates")
    sequence_id = seed.get("sequence")
    if not sequence_id:
        print("Seed has no sequence; downloading that image only.")
        return [(seed, 0.0)]

    data = get_collection(
        session, "image_ids", {"sequence_id": sequence_id}, max_pages=max_pages
    )
    ids = list(dict.fromkeys(str(item["id"]) for item in data if "id" in item))
    if str(image_id) not in ids:
        raise RuntimeError(
            "Seed image is absent from the fetched sequence. "
            "Try increasing --max-pages or using --image-id without --sequence."
        )
    center = ids.index(str(image_id))
    print(f"Sequence contains {len(ids)} fetched IDs; seed position is {center}.")

    # Check seed first, then alternating images before/after it in capture order.
    ordered_ids = [str(image_id)]
    for offset in range(1, len(ids)):
        before, after = center - offset, center + offset
        if before >= 0:
            ordered_ids.append(ids[before])
        if after < len(ids):
            ordered_ids.append(ids[after])
        if len(ordered_ids) >= max_scan:
            break

    selected = []
    for candidate_id in ordered_ids[:max_scan]:
        try:
            info = seed if candidate_id == str(image_id) else get_image(session, candidate_id)
            location = coords(info)
            if location is None:
                continue
            distance = distance_m(origin, location)
            if distance <= radius_m:
                selected.append((info, round(distance, 1)))
                if len(selected) >= limit:
                    break
        except (requests.RequestException, RuntimeError, ValueError) as error:
            print(f"Skipping metadata {candidate_id}: {type(error).__name__}")
    return selected


def parse_bbox(text):
    try:
        west, south, east, north = [float(x) for x in text.split(",")]
    except (ValueError, TypeError):
        raise ValueError("BBOX must be west,south,east,north") from None
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Invalid BBOX coordinate order or range")
    if (east - west) * (north - south) > 0.001:
        raise ValueError("BBOX is too large for this script; use a smaller area")
    return west, south, east, north


def bbox_images(session, bbox, limit, max_pages):
    west, south, east, north = parse_bbox(bbox)
    data = get_collection(
        session, "images",
        {"bbox": bbox, "fields": "id,computed_geometry", "limit": min(max(100, limit), 500)},
        max_pages=max_pages,
    )
    selected = []
    seen = set()
    for item in data:
        image_id = str(item.get("id", ""))
        if not image_id or image_id in seen:
            continue
        seen.add(image_id)
        try:
            info = get_image(session, image_id)
            location = coords(info)
            if location and west <= location[0] <= east and south <= location[1] <= north:
                selected.append((info, ""))
                if len(selected) >= limit:
                    break
        except (requests.RequestException, RuntimeError, ValueError) as error:
            print(f"Skipping metadata {image_id}: {type(error).__name__}")
    return selected


def download(session, info, distance, output):
    image_id = str(info["id"])
    image_url = info.get("thumb_2048_url") or info.get("thumb_1024_url")
    if not image_url:
        print(f"Skipping {image_id}: no thumbnail URL")
        return None

    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{image_id}.jpg"
    destination = images_dir / filename
    if not destination.exists() or destination.stat().st_size == 0:
        # Download with a separate request so the API token is not sent to the image CDN.
        with requests.get(image_url, timeout=60, stream=True) as response:
            response.raise_for_status()
            if not response.headers.get("Content-Type", "").lower().startswith("image/"):
                raise RuntimeError(f"Expected an image for {image_id}")
            temp_path = destination.with_suffix(".jpg.part")
            try:
                with temp_path.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            file.write(chunk)
                temp_path.replace(destination)
            finally:
                temp_path.unlink(missing_ok=True)
        print(f"Downloaded {filename}")
    else:
        print(f"Already exists: {filename}")

    point = coords(info)
    creator = info.get("creator") or {}
    if isinstance(creator, dict):
        creator = creator.get("username", "")
    return {
        "image_id": image_id,
        "filename": f"images/{filename}",
        "longitude": point[0] if point else "",
        "latitude": point[1] if point else "",
        "captured_at": info.get("captured_at", ""),
        "sequence_id": info.get("sequence", ""),
        "creator": creator,
        "compass_angle": info.get("compass_angle", ""),
        "distance_from_seed_m": distance,
        "source_url": f"https://www.mapillary.com/app/?pKey={image_id}",
        "license": "CC BY-SA (check attribution requirements)",
    }


def save_manifest(output, new_rows):
    path = output / "manifest.csv"
    records = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("image_id"):
                    records[row["image_id"]] = row
    for row in new_rows:
        records[row["image_id"]] = row
    temp_path = output / "manifest.csv.tmp"
    with temp_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in records.values():
            writer.writerow(row)
    temp_path.replace(path)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image-id", help="Known Mapillary image ID")
    mode.add_argument("--bbox", help="west,south,east,north (small area only)")
    parser.add_argument("--sequence", action="store_true", help="Select nearby frames from the seed's sequence")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--radius-m", type=float, default=250, help="Max distance from seed in meters")
    parser.add_argument("--max-scan", type=int, default=250, help="Maximum sequence frames to inspect")
    parser.add_argument("--max-pages", type=int, default=30, help="Max sequence or search API pages")
    parser.add_argument("--output", default="data/raw/mapillary/segment_001")
    args = parser.parse_args()

    if args.sequence and not args.image_id:
        parser.error("--sequence requires --image-id")
    if args.limit < 1 or args.max_scan < 1 or args.max_pages < 1 or args.radius_m <= 0:
        parser.error("--limit, --max-scan, --max-pages, and --radius-m must be positive")
    token = os.getenv("MAPILLARY_TOKEN")
    if not token:
        parser.error("Set MAPILLARY_TOKEN in your terminal before running")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    session = make_session(token)
    try:
        if args.image_id and args.sequence:
            selected = nearby_sequence_images(
                session, args.image_id, args.limit, args.radius_m,
                args.max_scan, args.max_pages,
            )
        elif args.image_id:
            selected = [(get_image(session, args.image_id), 0.0)]
        else:
            selected = bbox_images(session, args.bbox, args.limit, args.max_pages)

        print(f"Selected {len(selected)} images for download.")
        if not selected:
            print("No matches. Try a known --image-id with --sequence, or inspect another small BBOX.")
            return
        rows = []
        for info, distance in selected:
            try:
                row = download(session, info, distance, output)
                if row:
                    rows.append(row)
            except (requests.RequestException, RuntimeError, OSError) as error:
                print(f"Failed image {info.get('id')}: {type(error).__name__}: {error}")
        if rows:
            total = save_manifest(output, rows)
            print(f"Finished: {len(rows)} images present; {total} records in {output / 'manifest.csv'}")
        else:
            print("No files downloaded or verified.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
