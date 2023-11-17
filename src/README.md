### ml
Contains an [abstract base class](https://docs.python.org/3/library/abc.html) `Model` which provides an interface and supplementary method implementations for subclasses.
Additionally, contains implementations of `Model` for each machine learning method used in this project.
In general, these implementations provide abstractions for the intricacies of interacting with ML libraries implemented in Python or R.

### data.py
Contains code for loading and manipulating the loaded data (e.g., subset the data, or split it into training, testing, and validation sets).
