"""
Tests for the serving handler (model.py).

Calls on_startup() to load the model, then exercises handler() with
various inputs to verify output shape, schema, and basic correctness.
"""

import json

import pyarrow as pa

import model


def setup():
    """Load model once for all tests."""
    model.on_startup()


def test_handler_returns_utf8_array():
    """handler() should return a pa.Array of utf8 strings."""
    event = {"tweet": pa.array(["hello world"], type=pa.utf8())}
    result = model.handler(event, {})

    assert isinstance(result, pa.Array)
    assert result.type == pa.utf8()
    assert len(result) == 1


def test_handler_batch_size_matches_input():
    """Output array length should match input array length."""
    tweets = ["tweet one", "tweet two", "tweet three"]
    event = {"tweet": pa.array(tweets, type=pa.utf8())}
    result = model.handler(event, {})

    assert len(result) == len(tweets)


def test_handler_output_is_valid_json():
    """Each element in the output should be parseable JSON."""
    event = {"tweet": pa.array(["test tweet"], type=pa.utf8())}
    result = model.handler(event, {})

    for item in result.to_pylist():
        parsed = json.loads(item)
        assert isinstance(parsed, dict)


def test_handler_output_has_all_labels():
    """Each result should contain all 6 toxicity labels plus overall_toxic."""
    expected_keys = {
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
        "overall_toxic",
    }
    event = {"tweet": pa.array(["test tweet"], type=pa.utf8())}
    result = model.handler(event, {})

    parsed = json.loads(result[0].as_py())
    assert set(parsed.keys()) == expected_keys


def test_handler_probabilities_in_range():
    """Label probabilities should be between 0 and 1."""
    event = {"tweet": pa.array(["test tweet"], type=pa.utf8())}
    result = model.handler(event, {})

    parsed = json.loads(result[0].as_py())
    for label in ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"):
        assert 0.0 <= parsed[label] <= 1.0, f"{label} out of range: {parsed[label]}"


def test_handler_overall_toxic_is_bool():
    """overall_toxic should be a boolean."""
    event = {"tweet": pa.array(["test tweet"], type=pa.utf8())}
    result = model.handler(event, {})

    parsed = json.loads(result[0].as_py())
    assert isinstance(parsed["overall_toxic"], bool)


def test_handler_clean_vs_toxic():
    """A clearly clean tweet should score lower than a clearly toxic one."""
    event = {
        "tweet": pa.array(
            [
                "what a beautiful day, love this community!",
                "you're an absolute idiot, go die in a fire",
            ],
            type=pa.utf8(),
        )
    }
    result = model.handler(event, {})

    clean = json.loads(result[0].as_py())
    toxic = json.loads(result[1].as_py())
    assert toxic["toxic"] > clean["toxic"], (
        f"toxic tweet scored lower ({toxic['toxic']}) than clean ({clean['toxic']})"
    )


def test_handler_preprocesses_mentions_and_urls():
    """Tweets with @mentions and URLs should not crash the handler."""
    event = {
        "tweet": pa.array(
            [
                "@realuser123 check out https://example.com it's great!",
                "@someone @another hey look at http://t.co/abc",
            ],
            type=pa.utf8(),
        )
    }
    result = model.handler(event, {})

    assert len(result) == 2
    for item in result.to_pylist():
        parsed = json.loads(item)
        assert "toxic" in parsed


if __name__ == "__main__":
    setup()
    tests = [
        test_handler_returns_utf8_array,
        test_handler_batch_size_matches_input,
        test_handler_output_is_valid_json,
        test_handler_output_has_all_labels,
        test_handler_probabilities_in_range,
        test_handler_overall_toxic_is_bool,
        test_handler_clean_vs_toxic,
        test_handler_preprocesses_mentions_and_urls,
    ]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
