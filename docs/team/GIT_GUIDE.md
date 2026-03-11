# Git Guide

## Setup

See `TEAM_SETUP.md` for installing prerequisites, cloning the repo, and configuring your environment.

## Workflow

### 1. Pull latest main

```
git checkout main
git pull origin main
```

Do this every time before starting new work.

### 2. Create a branch

```
git checkout -b your-branch-name
```

One branch per task. Branch names: lowercase, hyphens, descriptive. Examples: `data-pipeline`, `cnn-model`, `lstm-model`, `evaluation-metrics`. Not: `fix`, `test`, `dev2`.

### 3. Work and commit

Stage and commit as you go. One logical change per commit.

```
git add .
git commit -m "feat: add mel-spectrogram extraction"
```

### 4. Push

```
git push -u origin your-branch-name
```

The `-u` flag is only needed on the first push for a new branch. After that, `git push` is enough.

### 5. Create a Pull Request

Go to GitHub. You'll see a prompt to create a PR from your branch. Create it. Name the PR clearly — the title becomes the merge commit message.

Ping the group chat so others know.

### 6. Merge

We use **Squash and merge**. This collapses all your branch commits into one clean commit on main. Select it from the merge button dropdown on GitHub.

After merging, delete the branch on GitHub (button appears right after merge). Then locally:

```
git checkout main
git pull origin main
git branch -d your-branch-name
```

Always delete merged branches. Don't leave them around.

## Commit Message Prefixes

Every commit message starts with a prefix:

| Prefix | When to use |
|--------|-------------|
| `feat:` | New functionality |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Restructuring code without changing behavior |
| `test:` | Adding or updating tests |
| `chore:` | Setup, config, dependencies, housekeeping |

If a commit includes a feature and its tests together, use `feat:`. The test is part of the feature delivery.

Examples of good messages:

```
feat: add MFCC extraction to preprocessing pipeline
fix: correct hop_length default in config
docs: add usage example to README
refactor: split dataset class into loader and transformer
test: add shape validation tests for mel-spectrogram output
chore: update torch version in pyproject.toml
```

Examples of bad messages:

```
update
fix stuff
WIP
asdf
changes
```

## Rules

1. **ALL work happens on a branch. No exceptions.** Never edit code on main. Always create a branch first, even for a one-line fix. If you realize you've been working on main, stop — stash your changes, create a branch, move them there.
2. **ALWAYS pull main before creating a new branch.** Every single time. If you skip this, your branch starts from outdated code and you'll get avoidable merge conflicts.
3. **Never merge broken code.** Your code must run without errors.
4. **Never commit data.** ESC-50, model checkpoints, `.npy` files, outputs — none of it goes in the repo.
5. **Always squash and merge.**
6. **Delete branches after merging.**
7. **Use your `@student.tuke.sk` email for all commits.**
8. **If two people need to edit the same file, take turns.** Coordinate in the group chat — one person works, pushes, the other pulls and continues. Don't work on the same file on separate branches at the same time unless you want to deal with merge conflicts.

## .gitignore

Already in the repo. Listed here for reference — don't remove any entries. If you need to add something, do it on a branch.

```
# Data
data/

# Model checkpoints and outputs
outputs/
*.pt
*.pth
*.npy

# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# OS (Windows)
Thumbs.db
desktop.ini

# OS (macOS)
.DS_Store
```

## If something goes wrong

**Before doing anything, describe the situation to your AI assistant.** Give it the full context — what you were trying to do, what happened, what error you see, what branch you're on. Let it walk you through the fix step by step. Don't guess your way through git recovery — one wrong command can make things worse.

**Undo last commit (keep changes):**

```
git reset --soft HEAD~1
```

**On the wrong branch:**

```
git stash
git checkout correct-branch
git stash pop
```

**Merge conflict:**

Git marks conflicting files. Open them, look for `<<<<<<<` markers, choose the right version, delete the markers. Then:

```
git add .
git commit -m "fix: resolve merge conflict in <filename>"
```

**Truly lost:**

Don't force-push. Don't delete things. Ask in the group chat.
