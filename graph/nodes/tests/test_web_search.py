from graph.nodes.web_search import _normalize_search_results


def test_normalize_search_results_accepts_string() -> None:
    assert _normalize_search_results("hello") == ["hello"]


def test_normalize_search_results_accepts_list_of_dicts() -> None:
    results = [{"content": "first"}, {"content": "second"}]
    assert _normalize_search_results(results) == ["first", "second"]


def test_normalize_search_results_accepts_dict_payload() -> None:
    results = {"results": [{"content": "one"}, {"content": "two"}]}
    assert _normalize_search_results(results) == ["one", "two"]
