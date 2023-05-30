Guide for Contributing to micromind
===================================

Step 0 - Preparing the Environment
----------------------------------

To ensure there are no conflicts between different library versions,
and that the CI pipeline works the same both locally and online,
it's recommended to create a dedicated environment for micromind.
After installing `Anaconda <https://www.anaconda.com>`_, you can achieve this with:

.. code-block:: shell

   conda create -n micromind python=3.8

Optionally, you can replace "micromind" with a name that you prefer for the environment.

Step 1 - Creating a Fork from the Original Repository
-----------------------------------------------------

Fork (and star! ⭐) the micromind repository to work on the project without any limitations
related to accesses to the official repository.


Step 2 - Installation
---------------------

After you have cloned the forked repository locally, navigate to the root folder and install
``micromind`` in editable mode, using the following command:

.. code-block:: shell

   pip install -e .[conversion]

Step 3 - Creating a New Branch
------------------------------

To contribute with new features in micromind, create a new branch and give it a significant name.
This might pertain to a new feature, patch, or bug fix that you're working on.

.. code-block:: shell

   git checkout -b your-branch-name

Step 4 - Implementing Changes
-----------------------------

On the new branch, unleash your creativity and commit all the changes modifications as you
normally would.

Step 5 - Unit tests and linters check
-------------------------------------

Linters
~~~~~~~
Before being merged, the code needs to pass unit tests and linters check.
To check if your modified codebase does so, you can install pre-commit hooks:

.. code-block:: shell

   pip install pre-commit

To configure pre-commit with the same settings decided for micromind, you should run

.. code-block:: shell

   pre-commit install

from inside the micromind root folder. pre-commit will check linters every time you
make a commit to the repo.

Unit tests
~~~~~~~~~~
To run unit tests, you should run

.. code-block:: shell

   pytest tests/

**Moreover, you should write additional tests if your contribution requires so.**

Step 6 - Making a Pull Request
------------------------------

Once your changes are complete, please contribute to the toolkit by creating a pull request.
Here are some guidelines for a good pull request, adapted from
`GitHub's blog post <https://github.blog/2015-01-21-how-to-write-the-perfect-pull-request/>`_:

* **Reason for the Pull Request**: clearly explain why you're making this pull request and what value it brings to the project;

* **Changes Made**: Describe the new behaviors, error fixes, or features that have been added.

By following these steps, you'll be well-equipped to make meaningful contributions to the project.
Happy coding!
