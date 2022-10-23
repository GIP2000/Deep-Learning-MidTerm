from CLIP_model import ClipHandler
from descriptor_test import LabelsWithDescriptors
from sys import argv

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

class Experiment:

    def __init__(self, labelDescriptList):
        self.labelDescriptList = labelDescriptList
        self.originalLabels = [l.labels[0] for l in labelDescriptList]
        self.descriporLabels = [d for d in l.descriptors for l in labelDescriptList]
        self.descripLabelMap = {d:l for d in l.descriptors for l in labelDescriptList}
        self.modelHandler = ClipHandler()


    #TODO Implement
    def run_original(self) -> float:
        self.modelHandler.setLabels(self.originalLabels)
        return 0


    #TODO Implement
    def run_descriptor(self) -> float:
        self.modelHandler.setLabels(self.descriporLabels)
        return 0


def main():
    exp = Experiment(get_label_descriptors())
    original_acc = exp.run_original()
    our_acc = exp.run_descriptor()

    print(f"CLIP base acc: {original_acc}\nOur acc: {our_acc}")

# This is the main program
if __name__ == '__main__':
    main()


