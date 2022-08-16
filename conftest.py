import os

import pytest
import tensorflow as tf


@pytest.fixture(scope="session")
def disable_traceback_filtering(request):
    # Allows us to disable traceback filtering at-will from the command line.
    if os.environ["DEBUG"]:
        tf.debugging.disable_traceback_filtering()
