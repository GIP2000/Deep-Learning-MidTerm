import torch
import clip
from PIL import Image
import numpy as np


class ClipHandler:

    def __init__(self, labels=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, self._preprocess = clip.load("ViT-B/32", device=self.device)
        self._labels = labels

    def _processImage(self, path:str):
        return self._preprocess(Image.open(path)).unsqueeze(0).to(self.device)

    def _processImageP(self, img):
        return self._preprocess(img).unsqueeze(0).to(self.device)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels:list[str]):
        self._labels = self._tokenize(labels)

    def _tokenize(self, labels: list[str]):
        return clip.tokenize(labels).to(self.device)


    def predict(self, imagePaths: list[str]):
        if self._labels is None:
            raise ValueError("Please Set the label values first")

        images = torch.tensor(np.concatenate([self._processImage(i) for i in imagePaths]), device=self.device)

        with torch.no_grad():

            logits_per_image, logits_per_text = self._model(images, self._labels)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        return probs

    def predictImages(self, X):
        if self._labels is None:
            raise ValueError("Please Set the label values first")

        images = torch.tensor(np.concatenate([self._processImageP(i) for i in X]), device=self.device)

        with torch.no_grad():

            logits_per_image, logits_per_text = self._model(images, self._labels)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        return probs



# Theses are the tests
if __name__ == "__main__":
    print("Starting")
    ch = ClipHandler()
    ch.labels = ["a diagram", "a dog", "a cat"]
    print(ch.predictBatch(["../data/clip.jpeg","../data/clip2.jpeg"]))
