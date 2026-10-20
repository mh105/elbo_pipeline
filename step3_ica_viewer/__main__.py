"""Reopen a prepared ICA viewer without rerunning preprocessing."""

import argparse
import threading

from .server import close_viewer, open_viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("viewer_dir", help="Prepared recording/fit viewer directory")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    url = open_viewer(args.viewer_dir, open_browser=not args.no_browser)
    print(f"ICA viewer: {url}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        close_viewer(args.viewer_dir)


if __name__ == "__main__":
    main()
