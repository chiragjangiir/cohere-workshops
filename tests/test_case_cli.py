from cohere_workshops.case_cli import build_parser


def test_accepts_required_case_metadata() -> None:
    options = build_parser().parse_args(
        [
            "/tmp/example",
            "--case-id",
            "example-pr-7",
            "--repository-url",
            "https://github.com/example/project.git",
            "--pull-request",
            "7",
            "--pre-change-commit",
            "a" * 40,
            "--post-change-commit",
            "b" * 40,
            "--change-request",
            "Update repository scanning.",
        ]
    )

    assert options.repository == "/tmp/example"
    assert options.case_id == "example-pr-7"
    assert options.pull_request == 7
    assert options.change_request == "Update repository scanning."