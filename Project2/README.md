# Project 2
The folder `cs-net` contains the source code for our python package where
the project required implementations are defined.

In the folder `notebook` this package are used to present the solutions of
the problems presented in the exercises.

To format the code there is added a [`pre-commit`](https://pre-commit.com)
configuration so the code follows a common standard between the members of
the group. After `pre-commit` is installed in virtual environment or
globally, activate it for this repository by running `pre-commit install`. It
will now run configured linters and formatters each time you make a commit.


## Setup using virtual environment

```console
# Create a virtual environment
python -m venv venv
# Activate it
venv\Scripts\activate.bat # or on linux/mac: . venv/bin/activate
# Install the package and dependencies as an editable package in the venv
pip install -e .[dev,testing]
```

## Running tests and check coverage

To run the `csnet` tests in `tests` folder we use `pytest` and `coverage`,
who is installed, if setup is done as described above.

```console
# to run tests:
(.venv)$ pytest
# to run coverage
(.venv)$ coverage run -m pytest && coverage report -m
```
