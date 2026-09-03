import os

os.environ.pop("SERPAPI_KEY", None)
os.environ.pop("FACE_SEARCH_PROVIDER_URL", None)
os.environ.pop("FACE_SEARCH_API_KEY", None)

import web_search
web_search.SERPAPI_KEY = None
web_search.FACE_SEARCH_PROVIDER_URL = None
web_search.FACE_SEARCH_API_KEY = None
os.environ.pop("OPENAI_API_KEY", None)

try:
    web_search.reverse_image_search("missing.png", query_encoding=[0.0])
except RuntimeError as exc:
    assert "Configure SERPAPI_KEY" in str(exc), exc
    print("Provider credential guard: PASS")
else:
    raise SystemExit("Provider credential guard failed")
