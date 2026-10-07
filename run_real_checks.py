"""Compatibility entry: use the same autonomous conversation as run_labweaver."""

from run_labweaver import main


if __name__ == "__main__":
    raise SystemExit(main(interactive=True))
