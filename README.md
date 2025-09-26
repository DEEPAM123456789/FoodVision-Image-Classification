# Food Image Classification – EfficientNet-B2 vs Vision Transformer (ViT)

A PyTorch-based food image classifier comparing **EfficientNet‑B2** and a **Vision Transformer (ViT)** baseline, with final deployment using EfficientNet‑B2 for a practical, lightweight inference footprint.

- **Notebook:** `09_pytorch_model_deployment.ipynb`
- **Dataset references: Food101, food-101, food101**

## Training & Setup
- **Epochs:** 10
- **Batch size:** 32
- **Learning rate:** 1e-3
- **Optimizer:** Adam

## Models
- **EfficientNet‑B2 (final deployment)** – chosen for smaller model size with fast startup and lower memory usage.
- **Vision Transformer (ViT)** – higher accuracy in experiments but **model size was >300 MB**, so not ideal for deployment constraints.

## Results
- **EfficientNet‑B2 best test accuracy:** `96.88%`
- **ViT (Vision Transformer) best test accuracy:** `98.47%`

## Deployment
The final deployment uses the **EfficientNet‑B2** checkpoint due to its smaller artifact size and faster cold starts. ViT weights (~300 MB) exceeded practical hosting limits and would significantly slow downloads on first run.

Typical deployment options:
- **Gradio** for a simple web UI .
- Export the trained PyTorch model via `torch.save()` or `torch.jit.trace/script` for reproducible inference.
- Package with `requirements.txt` and pin Torch & torchvision versions for consistent behavior.

## Inference (example)
```python
import torch
from PIL import Image
from torchvision import transforms

# load model
model = torch.load('efficientnet_b2_food_classifier.pth', map_location='cpu')
model.eval()

# preprocessing (adjust to your training transforms)
preproc = transforms.Compose([
    transforms.Resize((260, 260)),  # Example for EfficientNet-B2 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open('example.jpg').convert('RGB')
x = preproc(img).unsqueeze(0)  # shape: [1, C, H, W]
with torch.inference_mode():
    logits = model(x)
    prob = torch.softmax(logits, dim=1)
    pred = prob.argmax(dim=1).item()
print('Predicted class index:', pred)
```

## Why EfficientNet‑B2 for Deployment?
- **Model size & cold start:** ViT checkpoint was **319 MB**, leading to slower downloads and startup times, which degrades UX and may exceed hosting limits.
- **Resource usage:** Smaller models typically use less RAM/VRAM, enabling cheaper or free-tier hosting.
- **Accuracy trade‑off:** Although ViT had slightly better accuracy, **EfficientNet‑B2 achieved strong performance** while remaining deployment‑friendly.

## Reproducibility
- See the notebook for the full training pipeline, transforms, and evaluation steps.
- Set random seeds (`torch`, `numpy`, `random`) to make results more deterministic.
- Record your package versions in `requirements.txt` or `pip freeze > requirements.txt`.

## Future Work
- Mixed precision (`torch.cuda.amp`) to speed up training/inference on GPU.
- Model quantization (e.g., `torch.quantization` or ONNX Runtime) to further shrink the model.
- Knowledge distillation from ViT into a smaller student model.
- Augmentation tuning and class‑balanced sampling if the dataset is imbalanced.

## Credits
Built with **PyTorch**, **torchvision**, and community models (**EfficientNet‑B2**, **ViT**).
