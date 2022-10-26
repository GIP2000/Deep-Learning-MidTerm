from CLIP_model import ClipHandler
from descriptor_test import LabelsWithDescriptors
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
        # TODO make the has/is more general maybe use a CKF parse or something along those lines?
        return f"{cls} which (has/is) {descriptor}"

    @staticmethod
    def _make_label_from_class(cls:str) -> str:
        return f"a photo of a {cls}"


    # TODO Test & Vectorize
    def run_original(self, X: list[str], Y: list[str]) -> float:
        self.modelHandler.labels = self.originalLabels
        wrong = np.count_nonzero(self.modelHandler.predict(X).argmax(axis=1) - Y.T)
        return 1 - (float(wrong) / float(len(X)))


    # TODO Test & Vectorize
    def run_descriptor(self, X: list[str], Y: list[str]) -> float:
        self.modelHandler.labels = self.descriporLabels
        PHI = self.modelHandler.predict(X)
        print("prediction complete")
        wrong = np.count_nonzero((PHI @ self.descriptor_matrix).argmax(axis=1) - Y.T)
        return 1 - (float(wrong) / float(len(Y)))


def build_data(exp: Experiment, early_stop=None, img_per_class=1):
    test_folder = os.getcwd() + "/../data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/"
    skip_count = 0
    early_stop = 1000 if early_stop is None else early_stop
    X = []
    Y = []
    counter = 0
    for ld in tqdm(exp.labelDescriptList, desc="building_data", total=early_stop):
        folder_name = ld.folder
        x_l = os.listdir(test_folder + folder_name)[0:img_per_class]

        for x in x_l:
            X.append(test_folder + folder_name + "/" + x)
            Y.append(ld.index)

        if counter >= early_stop:
            break
        counter+=1

    assert(len(X) / img_per_class == early_stop)

    return np.array(X), np.array(Y)


def main():
    early_stop = None # so I can test it without running it on all of ImageNet

    exp = Experiment(get_label_descriptors(early_stop))
    # Sanity check
    assert(len(exp.labelDescriptList) ==1000 if early_stop is None else early_stop)

    X,Y = build_data(exp, early_stop)
    # Sanity check
    assert(len(np.unique(Y)) == 1000 if early_stop is None else early_stop)

    print("starting baseline CLIP model")
    original_acc = exp.run_original(X,Y)
    print("starting our test")
    our_acc = exp.run_descriptor(X,Y)

    print(f"CLIP base acc: {original_acc}\nOur acc: {our_acc}")

# This is the main program
if __name__ == '__main__':
    main()


