# Project Overview

This repository contains the source code for sunnypilot, a fork of comma.ai's openpilot, an open-source driver assistance system. It is designed to run on specific hardware in a car and provides features like adaptive cruise control and lane-keeping assist.

The project is primarily written in Python, with some parts in C/C++ for performance. It uses the SCons build system for C/C++ components and relies on a `pyproject.toml` file for Python dependencies.

# Building and Running

## Build

The project uses SCons for building the C/C++ parts. The main build configuration is in the `SConstruct` file. To build the project, you will likely need to run:

```bash
scons
```

This command will compile the necessary C/C++ files and generate the required libraries.

## Running

The main entry point for the application is `launch_openpilot.sh`, which in turn executes `launch_chffrplus.sh`. To start the application, you can run:

```bash
./launch_openpilot.sh
```

# Development Conventions

## Dependencies

Python dependencies are managed using `pyproject.toml`. To install them, you can use a tool like `pip`:

```bash
pip install -e .
```

The `-e` flag installs the project in editable mode, which is useful for development.

## Testing

The project uses `pytest` for testing. The configuration for pytest is in the `pyproject.toml` file. To run the tests, you can use the following command:

```bash
pytest
```

## Linting and Formatting

The project uses `ruff` for linting and formatting. The configuration is in the `pyproject.toml` file. You can run the linter with:

```bash
ruff check .
```

And the formatter with:

```bash
ruff format .
```
