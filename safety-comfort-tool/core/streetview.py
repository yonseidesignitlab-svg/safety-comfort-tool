"""
Google Street View helpers — metadata pre-check + raw image fetch.

Both functions are pure (no Streamlit). Mode A wraps them with progress bars;
Mode B drives them in a batch loop. Always validate panorama availability via
the (free) metadata endpoint BEFORE issuing a (paid) Street View Static call.
"""
import os
import time
import requests


STATIC_URL   = "https://maps.googleapis.com/maps/api/streetview"
METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"


def streetview_metadata_ok(lat, lng, api_key, *, use_outdoor=True, radius=50, timeout=5):
    """
    Returns True iff a usable panorama exists within `radius` meters of (lat, lng).

    The metadata endpoint is free; this prevents wasting Static-API quota on
    rural / interior / out-of-coverage points.
    """
    params = {"location": f"{lat},{lng}", "key": api_key, "radius": radius}
    if use_outdoor:
        params["source"] = "outdoor"
    try:
        r = requests.get(METADATA_URL, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json().get("status") == "OK"
    except requests.RequestException:
        pass
    return False


def fetch_streetview_image(
    lat, lng, heading, api_key,
    *, size_w=640, size_h=480, fov=90, pitch=0,
    use_outdoor=True, timeout=10, min_bytes=5000,
):
    """
    Issue one Street View Static request, return raw JPEG bytes or None.

    `min_bytes` guards against Google's "no imagery available" placeholder PNG
    that's typically <2 KB and a near-grey image.
    """
    params = {
        "size":     f"{int(size_w)}x{int(size_h)}",
        "location": f"{lat},{lng}",
        "heading":  str(heading),
        "pitch":    str(pitch),
        "fov":      str(fov),
        "key":      api_key,
    }
    if use_outdoor:
        params["source"] = "outdoor"
    try:
        r = requests.get(STATIC_URL, params=params, timeout=timeout)
        if r.status_code == 200 and len(r.content) > min_bytes:
            return r.content
    except requests.RequestException:
        pass
    return None


def save_image_bytes(content: bytes, dest_path: str) -> str:
    """Write JPEG bytes; create parent dirs if needed. Returns dest_path."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(content)
    return dest_path
