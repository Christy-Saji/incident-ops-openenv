"""Server entry point expected by openenv validate."""

import uvicorn

from app import app


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
