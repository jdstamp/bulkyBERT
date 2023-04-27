# bulkyBERT
CS 2470 project

## Setup

1. Install poetry: https://python-poetry.org/docs/
2. Clone repository: https://github.com/jdstamp/bulkyBERT
3. In the repository run `poetry install`

This should create a virtual environment and install all the declared dependencies.

## Adding or removing dependencies

Run `poetry add <dependency>` or `poetry remove <dependency>` to add or remove a dependency. 

## Workflow (assuming git >= 2.23)

To collaborate on a git repository, we need to work with branching. The main branch on the repository is `main`. This branch is protected for changes other than through pull requests on GitHub. The workflow will be as follows:

1. Make sure that your local repository has the latest changes to main:
    - `git switch main` 
    - `git fetch --prune`
    - `git pull`
2. Create a new branch from main: `git switch -c <new-branch>`
3. Make changes and run the dev dependency "black" with `poetry run black .` which will format all python files to be PEP8 compliant.
4. Stage changed files for commit: `git add <path(s)/to/changed/file(s)>`
5. Commit with a message that explains changes: `git commit -m"<reason for change>"`
6. Push to GitHub: `git push`. This command will not work the first time but tell you how to fix it to make it work, copy-paste and push.
7. Go to GitHub and create pull request to merge into `main`.
8. Assign reviewers to have them notified. 
9. Wait for reviews, address change requests, and eventually merge.


# References
- https://www.tensorflow.org/text/tutorials/transformer
