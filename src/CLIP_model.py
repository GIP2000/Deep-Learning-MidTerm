import torch
import clip
from PIL import Image


class ClipHandler:

    def __init__(self, labels=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, self._preprocess = clip.load("ViT-L/14@336px", device=self.device)
        self._labels = labels

    def _processImage(self, path:str):
        return self._preprocess(Image.open(path)).unsqueeze(0).to(self.device)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels:list[str]):
        self._labels = self._tokenize(labels)

    def _tokenize(self, labels: list[str]):
        return clip.tokenize(labels).to(self.device)

    def predict(self, imagePath: str):
        if self._labels is None:
            raise ValueError("Please Set the label values first")

        image = self._processImage(imagePath)

        with torch.no_grad():
            image_features = self._model.encode_image(image)
            text_features = self._model.encode_text(self._labels)

            logits_per_image, logits_per_text = self._model(image, self._labels)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        return probs

# Theses are the tests
if __name__ == "__main__":
    print("Starting")
    ch = ClipHandler()
    ch.labels = ["a diagram", "a dog", "a cat"]
    print(ch.predict("../data/clip.jpeg"))
