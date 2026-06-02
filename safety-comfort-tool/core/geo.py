"""
Geometry helpers for route/cell processing.

Sources merged:
- PREVIOUS/utils.py (slerp_point + carry-based resample_polyline)
- UPDATE/utils.py (cumulative-distance resample variant; bearing)
The PREVIOUS slerp version is preserved because Mode A (Streamlit) drew on it;
the UPDATE cumulative version was simpler but accumulated rounding drift on
long routes. The two yield equivalent results for spacings ≥ 5m.
"""
import math


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between two WGS-84 points, in meters."""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlmb  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def slerp_point(lat1, lon1, lat2, lon2, f):
    """Spherical interpolation at fraction f ∈ [0,1] of the great-circle arc."""
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r, lon2r = math.radians(lat2), math.radians(lon2)
    x1, y1, z1 = math.cos(lat1r) * math.cos(lon1r), math.cos(lat1r) * math.sin(lon1r), math.sin(lat1r)
    x2, y2, z2 = math.cos(lat2r) * math.cos(lon2r), math.cos(lat2r) * math.sin(lon2r), math.sin(lat2r)
    dot = max(-1.0, min(1.0, x1 * x2 + y1 * y2 + z1 * z2))
    omega = math.acos(dot)
    if omega == 0:
        return lat1, lon1
    so = math.sin(omega)
    t1 = math.sin((1 - f) * omega) / so
    t2 = math.sin(f * omega) / so
    x, y, z = t1 * x1 + t2 * x2, t1 * y1 + t2 * y2, t1 * z1 + t2 * z2
    return (
        math.degrees(math.atan2(z, math.sqrt(x * x + y * y))),
        math.degrees(math.atan2(y, x)),
    )


def resample_polyline(coords, spacing_m):
    """
    Resample a (lat, lng) polyline at fixed metric spacing using slerp.
    Always preserves the first and last vertex.
    """
    if len(coords) < 2:
        return coords[:]
    out = [coords[0]]
    carry = 0.0
    prev = coords[0]
    for i in range(1, len(coords)):
        curr = coords[i]
        seg = haversine_m(prev[0], prev[1], curr[0], curr[1])
        if seg == 0:
            continue
        d = spacing_m - carry
        while d <= seg:
            f = d / seg
            out.append(slerp_point(prev[0], prev[1], curr[0], curr[1], f))
            d += spacing_m
        carry = seg - (d - spacing_m)
        prev = curr
    if out[-1] != coords[-1]:
        out.append(coords[-1])
    return out


def calculate_polyline_length(coords):
    """Total polyline length in meters."""
    if len(coords) < 2:
        return 0.0
    return sum(
        haversine_m(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
        for i in range(len(coords) - 1)
    )


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Initial bearing from (lat1, lon1) to (lat2, lon2), 0–360°.
    North=0, East=90, South=180, West=270.
    """
    lat1r = math.radians(lat1)
    lon1r = math.radians(lon1)
    lat2r = math.radians(lat2)
    lon2r = math.radians(lon2)
    dlon  = lon2r - lon1r
    y = math.sin(dlon) * math.cos(lat2r)
    x = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360
