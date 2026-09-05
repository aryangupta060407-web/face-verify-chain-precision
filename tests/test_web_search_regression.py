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


if __name__ == "__main__":
    test_nested_provider_fields_are_normalized_and_downloadable()
    test_profile_candidate_is_ordered_before_product_candidate()
    print("web search regression tests: PASS")
