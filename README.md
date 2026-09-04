# Bachelor Machine Vision Project

This repository provides one CPU-only Python and OpenCV environment for the
entire bachelor group. The application runs inside Docker, so no local Python
installation is required on Windows, macOS, or Linux.

The first version intentionally uses headless OpenCV. It processes image files
and writes the result back to the host instead of opening GUI windows or using a
webcam.

## Prerequisites

- Windows or macOS: install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
- Linux: install Docker Engine and the Docker Compose plugin.
- Start Docker before running the commands below.

All commands are the same in PowerShell, macOS Terminal, and a Linux shell.

## Quick start

Build the image once after cloning the repository:

```sh
docker compose build
```

Place an image in `data/images/`, for example `data/images/input.jpg`, then run:

```sh
docker compose run --rm bachelor --input data/images/input.jpg --output data/output/edges.png
```

The processed image appears at `data/output/edges.png` on the host. Source,
tests, data, and model directories are bind-mounted, so changes to them are
visible in the container without rebuilding the image. Rebuild after changing
`requirements.txt` or `Dockerfile`.

Running `docker compose up` displays the CLI help and exits successfully.

## Development

Run the full test suite in the container:

```sh
docker compose run --rm --entrypoint pytest bachelor
```

Open a shell in the same environment:

```sh
docker compose run --rm --entrypoint sh bachelor
```

Show all CLI options:

```sh
docker compose run --rm bachelor --help
```

Stop and remove Compose containers and networks:

```sh
docker compose down
```

On Linux, the image uses user and group ID `1000` by default so generated files
are not owned by root. If your IDs differ, provide them while building:

```sh
APP_UID=$(id -u) APP_GID=$(id -g) docker compose build
```

## Project layout

```text
src/               Application code
  detection/       Detection modules
  tracking/        Tracking modules
  utils/           Shared helpers
data/              Bind-mounted input and output data
models/            Bind-mounted model weights
tests/             Automated tests
Dockerfile         Reproducible Python environment
compose.yaml       Cross-platform development commands
requirements.txt   Pinned Python dependencies
```

Datasets, generated output, and model weights are ignored by Git and excluded
from the Docker build context. Only `.gitkeep` placeholders are committed.

## Supported scope

The image supports CPU-based file processing on `linux/amd64` and
`linux/arm64`. CUDA, PyTorch, Ultralytics, Open3D, OpenCV GUI forwarding, and
live camera access are not included in this initial setup.
