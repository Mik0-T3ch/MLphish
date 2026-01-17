__version__ = "0.1.0"


# imports principales del paquete
# ojo: el archivo es data_loader.py, no dataloader.py
from .data_loader import DataLoader

# modelo principal
from .model import MLPhishModel

# trainer
# antes estaba en trainer.py pero al final quedó como train.py
from .train import Trainer

# evaluator
# archivo eval.py, no evaluator.py
from .eval import Evaluator

# utils
# importo la clase completa y no las funciones sueltas
from .utils import Utils


# exports públicos del paquete
__all__ = [
    "DataLoader",
    "MLPhishModel",
    "Trainer",
    "Evaluator",

    # dejo utils completo, no funciones individuales
    "Utils",

    # version del paquete
    "__version__",
]


# NOTA:
# falta implementar un archivo config o algo tipo load_config
# se deja comentado para después
#
# from .config import load_config
#
# cuando se agregue config, añadirlo también a __all__
