"""Create a small Git repository with an outdated and a current version."""

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

V1_PAYMENTS = '''"""Mobile money transfers."""


def calculate_transfer_fee(amount):
    """Return the fee for sending money. Flat 2 percent of the amount."""
    return round(amount * 0.02, 2)


def validate_phone_number(phone):
    """Check that a Kenyan phone number starts with 07 and has 10 digits."""
    return phone.startswith("07") and len(phone) == 10


def send_payment(phone, amount):
    """Send money to a phone number using the payments API."""
    if not validate_phone_number(phone):
        raise ValueError("Invalid phone number")
    fee = calculate_transfer_fee(amount)
    return {"to": phone, "amount": amount, "fee": fee, "api": "v1"}
'''

V2_PAYMENTS = '''"""Mobile money transfers."""


def calculate_transfer_fee(amount):
    """Return the fee for sending money using tiered pricing."""
    if amount <= 1000:
        return 0
    if amount <= 10000:
        return round(amount * 0.01, 2)
    return round(amount * 0.015, 2)


def normalize_phone_number(phone):
    """Convert a Kenyan phone number to international format (2547XXXXXXXX)."""
    digits = "".join(ch for ch in phone if ch.isdigit())
    if digits.startswith("0") and len(digits) == 10:
        return "254" + digits[1:]
    if digits.startswith("254") and len(digits) == 12:
        return digits
    raise ValueError("Invalid phone number")


def send_payment(phone, amount):
    """Send money to a phone number using the payments API."""
    recipient = normalize_phone_number(phone)
    fee = calculate_transfer_fee(amount)
    return {"to": recipient, "amount": amount, "fee": fee, "api": "v2"}
'''

RECEIPTS = '''"""Customer receipts."""


def format_receipt(phone, amount, fee):
    """Create a short text receipt for a completed transfer."""
    return f"Sent KES {amount} to {phone}. Fee: KES {fee}."
'''

VERSIONS = (
    ("v1", "v1: flat fee and phone validation", {
        "payments.py": V1_PAYMENTS,
        "receipts.py": RECEIPTS,
    }),
    ("v2", "v2: tiered fees and phone normalization", {
        "payments.py": V2_PAYMENTS,
        "receipts.py": RECEIPTS,
    }),
)


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        [
            "git",
            "-c", "user.name=Dunia Hub Workshop",
            "-c", "user.email=workshop@duniahub.xyz",
            "-c", "commit.gpgsign=false",
            "-c", "tag.gpgsign=false",
            *arguments,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )


def create_demo_repository(path: Path) -> Path:
    """Create the demo repository with tags v1 (outdated) and v2 (current)."""
    if path.exists():
        raise FileExistsError(f"{path} already exists")

    path.mkdir(parents=True)
    _git(path, "init", "--quiet")

    for tag, message, files in VERSIONS:
        for name, content in files.items():
            (path / name).write_text(content, encoding="utf-8", newline="\n")
        _git(path, "add", ".")
        _git(path, "commit", "--quiet", "-m", message)
        _git(path, "tag", tag)

    return path


def main(arguments: Sequence[str] | None = None) -> None:
    """Create the workshop demo repository."""
    parser = argparse.ArgumentParser(
        description="Create a demo Git repository with versions v1 and v2."
    )
    parser.add_argument("path", nargs="?", default="demo_repo")
    options = parser.parse_args(arguments)

    try:
        create_demo_repository(Path(options.path))
    except FileExistsError as error:
        print(f"{error}. Delete it to rebuild.", file=sys.stderr)
        raise SystemExit(1) from error

    print(f"Created {options.path} with tags v1 (outdated) and v2 (current).")


if __name__ == "__main__":
    main()