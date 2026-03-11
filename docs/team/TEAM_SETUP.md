# Team Setup

The repo is already created. Follow these steps to get your local environment ready.

## 1. Install Git

Check if you have it:

```
git --version
```

If not, download from https://git-scm.com/downloads. Run the installer, keep all defaults.

## 2. Install Python

Check if you have it:

```
python --version
```

If not, download from https://www.python.org/downloads/. During installation, **check "Add Python to PATH"**. This is critical — without it, `python` won't work in your terminal.

## 3. Install uv

```
pip install uv
```

Verify:

```
uv --version
```

## 4. GitHub Authentication

You need an SSH key to push and pull. If you don't have one set up, ask your AI assistant to walk you through generating an SSH key and adding it to your GitHub account. Give it your OS (Windows) and it'll give you the exact commands.

## 5. Clone the Repo

```
git clone git@github.com:<username>/<repo-name>.git
cd <repo-name>
```

## 6. Set Your Git Identity

```
git config user.name "Your Name"
git config user.email "your.name@student.tuke.sk"
```

Use your student TUKE email. This is mandatory — it tags every commit with your identity.

## 7. Install Dependencies

```
uv sync
```

This reads the lockfile and installs the exact same packages everyone else has. Run this every time you pull and see changes to `uv.lock`.

## 8. Verify Everything Works

```
git status
uv run python -c "import torch; print(torch.__version__)"
```

If both run without errors, you're ready. Read `GIT_GUIDE.md` before making any changes.

## Important: Always Use uv to Run Code

All Python commands in this project go through uv:

```
uv run python your_script.py
uv run pytest
```

Do **not** run `python your_script.py` or `pytest` directly. uv ensures that every command uses the same locked dependencies from `uv.lock`. Running without uv means your system Python or a different environment might pick up different package versions, and things that work on your machine will break on someone else's. Using uv keeps everyone on the exact same setup.
