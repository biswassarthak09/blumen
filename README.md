# blumen
Flower classification using IMAGENET50 and VISION TRANSFORMER


performance after running for 10 epochs

Resnet50
Epoch [1/10], Loss: 1.7086, Training Accuracy: 54.71%, Validation Accuracy: 40.29%
Epoch [2/10], Loss: 0.9558, Training Accuracy: 73.09%, Validation Accuracy: 66.76%
Epoch [3/10], Loss: 0.7146, Training Accuracy: 78.68%, Validation Accuracy: 84.12%
Epoch [4/10], Loss: 0.6746, Training Accuracy: 79.85%, Validation Accuracy: 82.94%
Epoch [5/10], Loss: 0.4442, Training Accuracy: 86.47%, Validation Accuracy: 90.59%
Epoch [6/10], Loss: 0.3409, Training Accuracy: 89.56%, Validation Accuracy: 93.24%
Epoch [7/10], Loss: 0.3496, Training Accuracy: 89.41%, Validation Accuracy: 92.35%
Epoch [8/10], Loss: 0.2733, Training Accuracy: 91.91%, Validation Accuracy: 93.24%
Epoch [9/10], Loss: 0.2030, Training Accuracy: 93.97%, Validation Accuracy: 93.82%
Epoch [10/10], Loss: 0.2050, Training Accuracy: 94.41%, Validation Accuracy: 94.41%


Vision Transformer
Epoch [1/10], Loss: 1.7567, Training Accuracy: 56.91%, Validation Accuracy: 91.76%
Epoch [2/10], Loss: 0.6382, Training Accuracy: 82.79%, Validation Accuracy: 89.12%
Epoch [3/10], Loss: 0.3794, Training Accuracy: 88.82%, Validation Accuracy: 95.88%
Epoch [4/10], Loss: 0.3170, Training Accuracy: 90.88%, Validation Accuracy: 95.29%
Epoch [5/10], Loss: 0.2156, Training Accuracy: 93.68%, Validation Accuracy: 94.41%
Epoch [6/10], Loss: 0.2859, Training Accuracy: 92.50%, Validation Accuracy: 95.59%
Epoch [7/10], Loss: 0.2502, Training Accuracy: 93.82%, Validation Accuracy: 96.18%
Epoch [8/10], Loss: 0.1327, Training Accuracy: 96.03%, Validation Accuracy: 98.24%
Epoch [9/10], Loss: 0.1144, Training Accuracy: 96.32%, Validation Accuracy: 98.82%
Epoch [10/10], Loss: 0.0817, Training Accuracy: 97.65%, Validation Accuracy: 98.82%


Test Accuracy: 98.24% (vit)

Vision Transformer performed better than the resnet50

Can be further optimized with HPO, ensembling, etc.