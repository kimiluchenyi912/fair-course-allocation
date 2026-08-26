from __future__ import annotations

import argparse

from .runner import validate_configuration


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate scheduling configuration files.")
    parser.add_argument("--config-dir", default="data/config")
    parser.add_argument("--templates-dir", default="data/templates")
    parser.add_argument(
        "--strict-policy",
        action="store_true",
        help="Treat current reference-baseline deviations as errors instead of warnings.",
    )
    args = parser.parse_args()

    report = validate_configuration(
        args.config_dir,
        args.templates_dir,
        strict_policy=args.strict_policy,
    )
    print(report.to_text())
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
