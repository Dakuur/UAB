# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:: Labs of Graphs, Topology, and Discrete Geometry - Data Engineering, UAB - 2019/2020 ::

Tools to help test assignments. 

NOTE: Students do not need to make direct use of this module.
"""

import os
import glob
import importlib
import hashlib
import pandas as pd
import tempfile


from test_all import options


class TestLab:
    """Utility class to allow multiple lab assignments from the same code base.
    """
    # e.g, "lab0"
    name = None
    # Dictionary to keep track of scores associated to each function
    fun_to_score = {}

    def __init__(self, *args, release_dir_path=None, **kwargs):
        if release_dir_path is not None:
            self._release_dir_path = release_dir_path
            
        super().__init__(*args, **kwargs)
        init_path = os.path.join(self.release_dir_path, "__init__.py")
        open(init_path, "w").write("""
import os
from os.path import dirname, basename, isfile, join
import glob
import importlib
modules_ = glob.glob(join(dirname(__file__), "*.py"))
base_module_name = basename(dirname(__file__))
for f in [f for f in modules_ if isfile(f) and not f.endswith('__init__.py')]:
    submodule_name = basename(f).replace(".py", "")
    globals()[submodule_name] = importlib.import_module(f"{base_module_name}.{submodule_name}", base_module_name)
\n""")

        self.loaded_module = importlib.import_module(os.path.basename(self.release_dir_path))

    def get_result_df(self):
        max_score = 0
        df = pd.DataFrame(columns=["fun", "max_score", "obtained_score", "note"])
        for fun, score in self.fun_to_score.items():
            assert score >= 0
            max_score += score
            fun = self.__getattribute__(fun.__name__)
            try:
                print(f"Evaluating function {fun.__name__} (score={score})...")
                signature_tests = get_dir_signature(os.path.dirname(__file__))
                signature_data = get_dir_signature(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))

                return_value = fun()

                assert signature_tests == get_dir_signature(os.path.dirname(__file__)), \
                    f"[WARNING] Lab {self.release_dir_path} modified the tests! " \
                    f"{(signature_tests, get_dir_signature(os.path.dirname(__file__)))}"
                assert signature_data == get_dir_signature(
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")), \
                    f"[WARNING] Lab {self.release_dir_path} modified the data!"

                if return_value is None:
                    return_value = 1
                if not 0.0 <= return_value <= 1.0:
                    raise ValueError(f"Test function {fun.__name__} yielded bogus "
                                     f"return value {return_value} not in [0,1]")
                df.loc[len(df)] = pd.Series(dict(fun=fun.__name__,
                                                 max_score=score,
                                                 obtained_score=round(score * return_value, 2),
                                                 note=f"OK ({100 * return_value:.1f}%)"))
            except (Exception, AssertionError) as ex:
                df.loc[len(df)] = pd.Series(dict(fun=fun.__name__,
                                                 max_score=score,
                                                 obtained_score=0,
                                                 note=repr(ex)))
                if not options.dont_exit_on_error:
                    raise ex

        return df

    @classmethod
    def score(cls, score):
        """Decorator to keep track of the score associated to each test function
        """
        assert score >= 0

        def decorator_fun(fun):
            assert fun not in cls.fun_to_score, \
                f"Redefining score for {fun} is not allowed. " \
                f"(previous: {cls.fun_to_score[fun]}"
            cls.fun_to_score[fun] = score
            return fun

        return decorator_fun

    @property
    def released_module(self):
        """Load the released module for this lab and user id
        """
        import_name = f"..{os.path.basename(self.release_dir_path)}"
        released_module = importlib.import_module(name=import_name, package=".")
        return released_module

    @property
    def release_dir_path(self):
        """Get the expected path of the release code for this lab and the group name
        """
        try:
            return self._release_dir_path
        except AttributeError:
            if self.name is None:
                raise NotImplementedError(f"Subclass {self.__class__.__name__} "
                                          f"should define the name attribute (not as None)")
            return os.path.join(os.path.dirname(os.path.dirname(__file__)), self.module_name)

    @property
    def module_name(self):
        try:
            return os.path.basename(self._release_dir_path)
        except AttributeError:
            return f"{self.name}_{options.id}"

    def test_folder_exists(self):
        assert os.path.exists(self.release_dir_path), \
            f"Expected directory {self.release_dir_path} does not exist.\n" \
            f"Please make sure you named your release folder and type the right NIU."

    def test_import(self):
        _ = self.released_module


def get_dir_signature(dir_path):
    """Get a SHA-based signature for the given dir
    """
    tuples = []
    for p in glob.glob(os.path.join(dir_path, "**", "*"), recursive=True):
        if "__pycache__" in p:
            continue
        try:
            hasher = hashlib.sha256()
            hasher.update(open(p, "rb").read())
            tuples.append((p, hasher.hexdigest()))
        except IsADirectoryError:
            tuples.append((p, "dir"))
    return '\n'.join(f"{t[0]}: {t[1]}" for t in sorted(tuples))
