import torch
from torchvision import transforms
from PIL import Image
from src.models.digital_reader import DigitalReader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DIGITAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def load_model(checkpoint_path: str = "models/checkpoints/digital_reader_best.pth") -> DigitalReader:
    model = DigitalReader().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

@torch.no_grad()
def predict_time_from_image(path: str, model: DigitalReader = None):
    if model is None:
        model = load_model()
    img = Image.open(path).convert("RGB")
    x = DIGITAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    out_h, out_m, out_s = model(x)
    h = out_h.argmax(dim=1).item()
    m = out_m.argmax(dim=1).item()
    s = out_s.argmax(dim=1).item()
    return h, m, s

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict_digital_time <path_to_image>")
        sys.exit(1)
    img_path = sys.argv[1]
    model = load_model()
    h, m, s = predict_time_from_image(img_path, model)
    print(f"Predicted time: {h:02d}:{m:02d}:{s:02d}")
