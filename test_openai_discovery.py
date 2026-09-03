import os
from types import SimpleNamespace

os.environ.pop("OPENAI_API_KEY", None)

from openai_discovery import _annotation_values, discover

annotation = SimpleNamespace(type="url_citation", url="https://example.org/post", title="Example post")
content = SimpleNamespace(annotations=[annotation])
output = SimpleNamespace(content=[content])
response = SimpleNamespace(output=[output])

values = _annotation_values(response)
assert values == [{"url": "https://example.org/post", "title": "Example post"}]
assert discover("missing.png") == []
print("OpenAI discovery parser/no-key test: PASS")
