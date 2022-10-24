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
            lst = LabelsWithDescriptors.create_descriptors_from_label_file(f, early_stop)
            with open(os.getcwd() + "/../data/new_cats.json", "w+") as outfile:
                json.dump([x for x in lst], outfile, cls=LabelsWithDescriptors.MyEncoder)
            return lst
        return LabelsWithDescriptors.read_list_from_file(f, early_stop)


def build_data(early_stop=None):
    image_folders = os.getcwd() + "/../data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/"
    skip_count = 0
    X = []
    Y = []
    for (i, (root, dirs, files)) in tqdm(enumerate(os.walk(image_folders, topdown=True)), desc="building_data", total=1000 if early_stop is None else early_stop):
        if len(files) <= 0 or root == image_folders:
            skip_count += 1
            continue

        # This is a sanity check can be removed for performance in the future
        assert(i - skip_count >= 0 and i - skip_count <= 1000)

        # Allows testing on smaller subset
        if early_stop is not None and i > early_stop:
            break

        # print(f"starting class: {i - skip_count}")
        # print(f"starting class: {i - skip_count}\n root: {root}\ndirs: {dirs}\nfiles:{files}")

        for x in files:
            if ".DS_Store" in x:
                continue
            X.append(root + "/" + x)
            Y.append(i - skip_count)

    return np.array(X), np.array(Y)


class Experiment:

    def __init__(self, labelDescriptList):
        self.labelDescriptList = labelDescriptList
        self.originalLabels = np.array([Experiment._make_label_from_class(l.labels[0]) for l in labelDescriptList])
        self.descriporLabels = np.array([ Experiment._combine_label_and_descriptor(l.labels[0], d) for l in labelDescriptList for d in l.descriptors])
        self.descripLabelMap = {Experiment._combine_label_and_descriptor(l.labels[0], d):l for l in labelDescriptList for d in l.descriptors}
        self.modelHandler = ClipHandler()

    @staticmethod
    def _combine_label_and_descriptor(label: str, descriptor: str) -> str:
        # TODO make the has/is more general maybe use a CKF parse or something along those lines?
        return f"{label} which (has/is) {descriptor}"

    @staticmethod
    def _make_label_from_class(cls:str) -> str:
        return f"a {cls}"


    # TODO Test & Vectorize
    def run_original(self, X: list[str], Y: list[str]) -> float:
        self.modelHandler.labels = self.originalLabels
        num_correct = 0
        pbar = tqdm(enumerate(zip(X, Y)), total=len(X))
        for (i,(x, y)) in pbar:
            if self.modelHandler.predict(x).argmax() == y:
                num_correct += 1
            pbar.set_description(f"original acc: {float(num_correct) / float(i+1)}")

        return float(num_correct) / float(len(X))


    # TODO Test & Vectorize
    def run_descriptor(self, X: list[str], Y: list[str]) -> float:
        self.modelHandler.lables = self.descriporLabels
        num_correct = 0
        pbar = tqdm(enumerate(zip(X, Y), desc="Our Predictions: ", total=len(X)))
        for (i, (x, y)) in pbar:
            label_prob = np.zeros(len(self.originalLabels));
            PHI = self.modelHandler.predict(x)
            for (phi, dl) in zip(PHI, self.descriporLabels):
                label_prob[self.descripLabelMap[dl].index] += phi

            for (lp,l) in zip(label_prob,self.labelDescriptList):
                lp /= len(l.descriptors)
            if label_prob.argmax() == y:
                num_correct += 1

            pbar.set_description(f"Our Acc: {float(num_correct) / float(i+1)}")

        return float(num_correct) / float(len(Y))


def main():
    early_stop = 5
    exp = Experiment(get_label_descriptors(early_stop))
    X,Y = build_data(early_stop)

    # Sanity check
    assert(len(np.unique(Y)) == early_stop)

    original_acc = exp.run_original(X,Y)

    # our_acc = exp.run_descriptor(X,Y)
    our_acc = 0

    print(f"CLIP base acc: {original_acc}\nOur acc: {our_acc}")

# This is the main program
if __name__ == '__main__':
    main()


