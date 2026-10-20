"""A model driver package kept outside the repository.

tests/replay/test_external_driver.py copies it into a temporary directory and puts that directory
on sys.path, as installing it into the run's environment would; it is never imported from here.
"""
