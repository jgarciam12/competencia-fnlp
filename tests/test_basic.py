import os
import pandas as pd
import sklearn

def test_environment_packages():
    # Comprueba que pandas y scikit-learn están disponibles
    assert hasattr(pd, "__version__")
    assert hasattr(sklearn, "__version__")

def test_repo_files_exist():
    # Comprueba que existen archivos clave del repo
    repo_root = os.path.dirname(os.path.dirname(__file__)) # .. = raiz del repo

    assert os.path.exists(os.path.join(repo_root, "src", "train.py"))
    assert os.path.exists(os.path.join(repo_root, "src", "predict.py"))