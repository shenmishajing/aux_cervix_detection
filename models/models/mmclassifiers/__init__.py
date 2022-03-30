from .heads import *
from .image import ImageClassifier
from .image_dimly import ImageDimlyClassifier
from .image_distillation_label import ImageDistillationLabelClassifier, ImageDistillationLabelTeacherClassifier
from .image_learn_label import ImageLearnLabelClassifier
from .image_rotate import RotateImageClassifier
from .image_transformer import ImageTransformerClassifier
from .image_transformer_distillation_label import ImageTransformerDistillationLabelClassifier, \
    ImageTransformerDistillationLabelTeacherClassifier
from .image_transformer_with_label import ImageTransformerWithLabelClassifier
from .image_with_label import ImageWithLabelClassifier
from .image_with_label_no_fusion import ImageWithLabelNoFusionClassifier
from .label import LabelClassifier
from .losses import *
from .mmcls_adapter import MMClsModelAdapter
from .necks import *
