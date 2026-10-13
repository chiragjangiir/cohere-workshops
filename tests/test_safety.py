from cohere_workshops.safety import detect_secret_types


def test_detects_high_confidence_secret_patterns() -> None:
    content = "\n".join(
        (
            "-----BEGIN " + "PRIVATE KEY-----",
            "AKIA" + "IOSFODNN7EXAMPLE",
            "ghp_" + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
            "AIza" + ("A" * 35),
        )
    )

    assert detect_secret_types(content) == (
        "private_key",
        "aws_access_key",
        "github_token",
        "google_api_key",
    )


def test_does_not_flag_normal_source_code() -> None:
    content = """
def load_api_key() -> str:
    return os.environ["API_KEY"]
"""

    assert detect_secret_types(content) == ()
