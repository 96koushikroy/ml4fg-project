# Modeling Chromatin Insulator Loops with Long-Range Transformers

## Abstract
Chromatin insulator loops play a critical role in regulating gene expression by
physically isolating genes from nearby promoter regions. Mutations that undo
loop formation can therefore increase the risk of disease. Previous approaches
to modeling insulator loops have leveraged convolutional and recurrent neural
network architectures. However, attention-based models, such as transformers,
have demonstrated state of the art performance for many sequence modeling tasks,
including in genomics and structural biology. We apply two transformer-based
models, DNABERT and the Enformer, to identify anchor regions of chromatin
loops. We find that these models are more effective at distinguishing true anchors
from similar non-anchor regions.

## Code Overview:
- `boundary_models.py`: contains CNN and LSTM models/modules
- `boundary_transformers.py`: contains transformer models
- `train_boundary.py`: training loop for models
- `train_boundary_[MODEL].py`: driver scripts for training different model types
- `evaluate.py`: runs model on all test sets