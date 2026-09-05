"""Regression coverage for provider normalization and mixed-result ordering."""
import web_search


def test_nested_provider_fields_are_normalized_and_downloadable():
    raw = {
        "title": "Public profile photo",
        "url": {"url": "https://x.com/example/status/1"},
        "image": {"original": {"url": "https://img.example/profile.jpg"}},
        "thumbnail": {"src": "https://img.example/thumb.jpg"},
    }
    normalized = web_search._normalise(raw, "Google Lens")
    assert normalized["link"] == "https://x.com/example/status/1"
    assert normalized["image"] == "https://img.example/profile.jpg"
    assert normalized["thumbnail"] == "https://img.example/thumb.jpg"

    original_download = web_search._download
    original_page_image = web_search._page_image
    try:
        web_search._download = lambda url, referer=None: b"candidate-bytes" if "profile.jpg" in url else None
        web_search._page_image = lambda _: None
        assert web_search._candidate_image(raw) == b"candidate-bytes"
    finally:
        web_search._download = original_download
        web_search._page_image = original_page_image


def test_profile_candidate_is_ordered_before_product_candidate():
    person = web_search._normalise(
        {"title": "Public profile photo", "link": "https://x.com/example/status/1", "image": "https://img/profile.jpg"},
        "Google Lens",
    )
    product = web_search._normalise(
        {"title": "Calvin Klein clothing product", "link": "https://shop.example/product/123", "image": "https://img/product.jpg"},
        "Google Lens",
    )
    assert web_search._candidate_order_key(person) > web_search._candidate_order_key(product)


def test_targeted_search_expands_public_platforms_and_preserves_provenance():
    calls = []
    original_serpapi = web_search._serpapi
    try:
        def fake_serpapi(params):
            calls.append(params)
            return {"organic_results": [{
                "title": "Rashmit public profile",
                "link": "https://instagram.com/rashmit.example",
                "snippet": "Public profile photo",
                "displayed_link": "instagram.com/rashmit.example",
                "image": {"url": "https://img.example/rashmit.jpg"},
            }]}
        web_search._serpapi = fake_serpapi
        results = web_search._google_context_search(["rashmit"], max_per_query=2)
    finally:
        web_search._serpapi = original_serpapi
    assert len(calls) == 10
    assert {"x.com", "instagram.com", "linkedin.com", "facebook.com", "pinterest.com"} == {
        query["q"].split("site:", 1)[1].split()[0] for query in calls
    }
    assert results[0]["snippet"] == "Public profile photo"
    assert results[0]["search_query"] == calls[0]["q"]
    assert results[0]["image"] == "https://img.example/rashmit.jpg"


def test_google_lens_parser_captures_documented_and_variant_sections():
    fixture = {
        "exact_matches": [{
            "title": "Exact public photo",
            "link": "https://example.com/exact",
            "source": "Example",
            "thumbnail": "https://img.example/exact-thumb.jpg",
            "image": "https://img.example/exact.jpg",
        }],
        "image_sources": [{
            "title": "Older source page",
            "link": "https://example.com/source",
            "source": "Example Archive",
            "thumbnail": "https://img.example/source-thumb.jpg",
        }],
        "visual_results": [{
            "title": "Variant visual result",
            "link": "https://example.com/visual",
            "source": "Example Visual",
            "thumbnail": "https://img.example/visual-thumb.jpg",
            "image": "https://img.example/visual.jpg",
        }],
    }
    exact = web_search._lens_records(fixture, "exact_matches", "exact_results", "image_sources")
    visual = web_search._lens_records(fixture, "visual_matches", "visual_results")
    assert len(exact) == 2
    assert len(visual) == 1
    normalized = [web_search._normalise(item, "Google Lens", True) for item in exact]
    assert [item["link"] for item in normalized] == [
        "https://example.com/exact",
        "https://example.com/source",
    ]
    assert normalized[0]["image"] == "https://img.example/exact.jpg"


def test_google_lens_uses_explicit_types_and_native_image_id():
    original_serpapi = web_search._serpapi
    calls = []
    try:
        def fake_serpapi(params):
            calls.append(params)
            if params["type"] == "exact_matches":
                return {"exact_matches": [{"title": "exact", "link": "https://example.com/e"}]}
            return {"visual_matches": [{"title": "visual", "link": "https://example.com/v"}]}
        web_search._serpapi = fake_serpapi
        exact, visual = web_search._google_lens("native-image-id", use_image_id=True)
    finally:
        web_search._serpapi = original_serpapi
    assert len(exact) == 1
    assert len(visual) == 1
    assert calls == [
        {"engine": "google_lens", "image_id": "native-image-id", "type": "exact_matches"},
        {"engine": "google_lens", "image_id": "native-image-id", "type": "visual_matches"},
    ]


if __name__ == "__main__":
    test_nested_provider_fields_are_normalized_and_downloadable()
    test_profile_candidate_is_ordered_before_product_candidate()
    test_targeted_search_expands_public_platforms_and_preserves_provenance()
    test_google_lens_parser_captures_documented_and_variant_sections()
    test_google_lens_uses_explicit_types_and_native_image_id()
    print("web search regression tests: PASS")
