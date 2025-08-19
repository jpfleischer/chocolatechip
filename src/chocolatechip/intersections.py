
intersection_lookup = {
    3287: "Stirling Road and N 68th Avenue",
    3248: "Stirling Road and N 66th Avenue",
    3032: "Stirling Road and SR-7",
    3265: "Stirling Road and University Drive",
    3334: "Stirling Road and SW 61st Avenue",
    3252: "Stirling Road and Davie Road Extension",
    5060: "SW 13th Street and University Avenue",
}

cam_lookup = {
    3287: 24,
    3248: 27,
    3032: 23,
    3265: 30,
    3334: 33,
    3252: 36,
    5060: 7
}


IMAGE_BASE_URL = "http://maltlab.cise.ufl.edu:30101/api/image/"

# STRICT mapping (only numbered map files; no resized/PO/custom names)
# (Assuming 36 -> "36_Map.png", as present in your directory listing.)
CAMERA_IMAGE = {
    7:  "07_Map.jpg",
    24: "24_Map.png",
    27: "27_Map.png",
    23: "21_Map.png",
    30: "30_Map.png",
    33: "33_Map.png",
    36: "36_Map.png",
}

# Convenience: list of all pictures to prefetch
PICTURES = sorted(set(CAMERA_IMAGE.values()))

def map_image_for(intersection_id: int, camera_id: int | None = None) -> str:
    """
    Return the exact filename for the given intersection/camera.
    No heuristic fallback—if it's not in CAMERA_IMAGE, we raise.
    """
    if camera_id is None:
        camera_id = cam_lookup.get(intersection_id)
    if camera_id is None:
        raise KeyError(f"No camera_id for intersection_id={intersection_id}")
    try:
        return CAMERA_IMAGE[int(camera_id)]
    except KeyError:
        raise KeyError(f"No image mapping for camera_id={camera_id}")