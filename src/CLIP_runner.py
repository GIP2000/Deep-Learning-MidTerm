from CLIP_model import ClipHandler
from descriptor_test import LabelsWithDescriptors
from sys import argv
import numpy as np

def get_label_descriptors():
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
            lst = LabelsWithDescriptors.create_descriptors_from_label_file(f)
            with open(os.getcwd() + "../data/new_cats.json", "w") as outfile:
                json.dump([x for x in lst], outfile, cls=LabelsWithDescriptors.MyEncoder)
            return lst
        return LabelsWithDescriptors.read_list_from_file(f)


#TODO implement
def build_data():

    pass


class Experiment:

    def __init__(self, labelDescriptList):
        self.labelDescriptList = labelDescriptList
        self.originalLabels = np.array([l.labels[0] for l in labelDescriptList])
        self.descriporLabels = np.array([ Experiment._combine_label_and_descriptor(d) for d in l.descriptors for l in labelDescriptList])
        self.descripLabelMap = {Experiment._combine_label_and_descriptor(d):l for d in l.descriptors for l in labelDescriptList}
        self.modelHandler = ClipHandler()

    @staticmethod
    def _combine_label_and_descriptor(label: str, descriptor: str) -> str:
        # TODO make the has/is more general maybe use a CKF parse or something along those lines?
        return f"{label} which (has/is) {descriptor}"


    # TODO Test & Vectorize
    def run_original(self, X: list[str], Y: list[str]) -> float:
        self.modelHandler.labels = self.originalLabels
        num_correct = 0
        for (x, y) in zip(X, Y):
            if self.originalLabels[self.modeHandler.predict(x).argmax()] == y:
                num_correct += 1

        return float(num_correct) / float(len(image_paths))


    # TODO Test & Vectorize
    def run_descriptor(self, X: list[str], Y: list[str]) -> float:
        self.modelHandler.lables = self.descriporLabels
        num_correct = 0

        for (x, y) in zip(X, Y):
            label_prob = np.zeros(len(self.originalLabels));
            PHI = self.modelHandler.predict(x)
            for (phi, dl) in zip(PHI, self.descriporLabels):
                label_prob[self.descripLabelMap[dl].index] += phi

            for (lp,l) in zip(label_prob,self.labelDescriptList):
                lp /= len(l.descriptors)
            if self.originalLabels[label_prob.argmax()] == y:
                num_correct += 1

        return num_correct


def main():
    exp = Experiment(get_label_descriptors())
    X,Y = build_data()
    original_acc = exp.run_original(X,Y)
    our_acc = exp.run_descriptor(X,Y)

    print(f"CLIP base acc: {original_acc}\nOur acc: {our_acc}")

# This is the main program
if __name__ == '__main__':
    main()


