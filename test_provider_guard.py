import os

os.environ.pop("SERPAPI_KEY", None)
os.environ.pop("FACE_SEARCH_PROVIDER_URL", None)
os.environ.pop("FACE_SEARCH_API_KEY", None)

from web_search import reverse_image_search

try:
    reverse_image_search("missing.png", query_encoding=[0.0])
except RuntimeError as exc:
    assert "Configure SERPAPI_KEY" in str(exc), exc
    print("Provider credential guard: PASS")
else:
    raise SystemExit("Provider credential guard failed")
