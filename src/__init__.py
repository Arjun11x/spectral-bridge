# makes src/ a Python package — required for all cross-file imports.
# importing core symbols here allows shorthand usage from notebooks:
#
#   from src import SpectralTransformer
#   from src import train, evaluate, predict
#
# instead of the longer:
#
#   from src.model import SpectralTransformer
#   from src.train import train

from src.model   import SpectralTransformer, build_model, masked_mse_loss
from src.train   import train
from src.evaluate import evaluate
from src.predict  import predict

