from CLIP_model import ClipHandler
from descriptor_test import LabelsWithDescriptors
from datasets import load_dataset
import os
from sys import argv
import json
import numpy as np
from tqdm import tqdm

def get_label_descriptors(early_stop=None):
    if len(argv) != 3:
        print("Usage: python CLIP_runner.py -[g|r] [file_path]")
        exit(1)
    elif argv[1] != "-g" and argv[1] != "-r":
        print("Usage: python CLIP_runner.py -[g|r] [file_path]")
        exit(2)

    [_, flag, pathR] = argv
    path = os.getcwd() + "/" + pathR
    with open(path) as f:
        if flag == "-g":
            out_path = os.getcwd() + "/../data/new_cats.json"
            lst = LabelsWithDescriptors.create_descriptors_from_label_file(f, early_stop, out_path)
            return lst
        return LabelsWithDescriptors.read_list_from_file(f, early_stop)




class Experiment:

    def __init__(self, labelDescriptList):
        self.labelDescriptList = labelDescriptList
        self.folderIndexMap = {l.folder: l.index for l in labelDescriptList}
        self.originalLabels = np.array([Experiment._make_label_from_class(l.labels[0]) for l in labelDescriptList])
        self.descriporLabels = np.array([ Experiment._combine_label_and_descriptor(l.labels[0], d) for l in labelDescriptList for d in l.descriptors])
        self.descriptor_matrix = np.zeros((len(self.descriporLabels), len(self.originalLabels)))
        offset = 0
        for (ci,l) in enumerate(labelDescriptList):
            val = 1/len(l.descriptors)
            for _ in l.descriptors:
                self.descriptor_matrix[offset][ci] = val
                offset += 1
        self.modelHandler = ClipHandler()

    @staticmethod
    def _combine_label_and_descriptor(cls: str, descriptor: str) -> str:
        # TODO make the has/is more general maybe use a CKY parse or something along those lines?
        return f"a phot of a {cls}, which (has/is) {descriptor}"

    @staticmethod
    def _make_label_from_class(cls:str) -> str:
        return f"a photo of a {cls}"


    # TODO Test & Vectorize
    def run_original(self, X: list[str], Y:np.ndarray ) -> float:
        self.modelHandler.labels = self.originalLabels
        wrong = np.count_nonzero(self.modelHandler.predict(X).argmax(axis=1) - Y.T)
        return 1 - (float(wrong) / float(len(X)))

    def run_originalP(self, X,  Y: np.ndarray ) -> float:
        self.modelHandler.labels = self.originalLabels
        wrong = np.count_nonzero(self.modelHandler.predictImages(X).argmax(axis=1) - Y.T)
        return 1 - (float(wrong) / float(len(X)))

    # TODO Test & Vectorize
    def run_descriptor(self, X: list[str], Y: np.ndarray ) -> float:
        self.modelHandler.labels = self.descriporLabels
        PHI = self.modelHandler.predict(X)
        print("Prediction Complete Performing Matrix Multiply")
        wrong = np.count_nonzero((PHI @ self.descriptor_matrix).argmax(axis=1) - Y.T)
        return 1 - (float(wrong) / float(len(Y)))


    def run_descriptorP(self, X, Y: np.ndarray ) -> float:
        # self.modelHandler.labels = self.descriporLabels
        # PHI = self.modelHandler.predictImages(X)
        # print("Prediction Complete Performing Matrix Multiply")
        wrong = np.count_nonzero(self.get_predictionsP(X)[0] - Y.T)
        return 1 - (float(wrong) / float(len(Y)))

    def get_predictionsP(self, X) -> np.ndarray:
        self.modelHandler.labels = self.descriporLabels
        PHI = self.modelHandler.predictImages(X)
        return (PHI @ self.descriptor_matrix).argmax(axis=1), PHI



def build_data(exp: Experiment, img_count=None):
    dataset = load_dataset("imagenet-1k", use_auth_token=True, split="validation").shuffle(seed=31415)
    X = []
    Y = []
    for (i,data) in tqdm(enumerate(dataset), total=img_count if img_count is not None else dataset.num_rows):
        if data["label"] >= len(exp.labelDescriptList):
            continue
        if len(X) >= img_count:
            break
        X.append(data["image"])
        Y.append(data["label"])
    return X, np.array(Y)

def main():
    early_stop = None # so I can test it without running it on all of ImageNet
    img_count = 1000

    exp = Experiment(get_label_descriptors(early_stop))
    # Sanity check
    assert(len(exp.labelDescriptList) ==1000 if early_stop is None else early_stop)

    X,Y = build_data(exp, img_count)
    assert(early_stop is None and len(X) == img_count)


    print("starting baseline CLIP model")
    original_acc = exp.run_originalP(X,Y)
    print("starting our test")
    our_acc = exp.run_descriptorP(X,Y)

    print(f"CLIP base acc: {original_acc}\nOur acc: {our_acc}")

# This is the main program
if __name__ == '__main__':
    main()


