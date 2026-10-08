"""D-FINE-X inference components vendored from ASCENT's D-FINE checkout.

Only register inference components: importing training datasets, profilers,
or servers would introduce unnecessary dependencies into LCN evaluation.
"""

from .nn.backbone.hgnetv2 import HGNetv2
from .zoo.dfine import DFINE, DFINEPostProcessor, DFINETransformer, HybridEncoder
