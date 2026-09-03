import os

os.environ.pop("OPENAI_API_KEY", None)

from ai_evidence import explain_evidence

result = explain_evidence([
    {
        "link": "https://example.org/post",
        "title": "Authorized post",
        "source": "example.org",
        "engine": "test",
        "face_verified": False,
        "face_similarity": 0.21,
    }
])
assert result["model_used"] is False, result
assert result["verified_candidate_count"] == 0, result
assert "No candidate" in result["summary"], result
print("AI evidence smoke test: PASS")
