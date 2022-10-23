import openai
import json
import pathlib
from json import JSONEncoder
import os
from sys import argv

openai.api_key = os.getenv("OPEN_AI_API_KEY")

class LabelsWithDescriptors:

    CLASS_NAME = "LabelsWithDescriptors"


    def __init__(self, index, labels, descriptors=None):
        self.index = index
        self.labels = labels
        if descriptors is not None:
            self.descriptors = descriptors
        else:
            self.descriptors = LabelsWithDescriptors._get_parsed_response(self.labels[0])

    def __str__(self):
        return f"#{self.index}: {self.labels} -> {self.descriptors}"

    @staticmethod
    def _build_input(category_name: str) -> str:
        return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a
     photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a
     photo:
-"""


    @staticmethod
    def _get_response(input: str):
        return openai.Completion.create(
          model="text-davinci-002",
          prompt=LabelsWithDescriptors._build_input(input),
          temperature=0.7,
          max_tokens=100,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )


    @staticmethod
    def _get_parsed_response(input:str):
        raw = LabelsWithDescriptors._get_response(input).get("choices")[0]["text"]
        return [x[1:].strip() for x in raw.split("\n")]


    @staticmethod
    def read_list_from_file(file_path: str):
        return json.load(file_path, object_hook=LabelsWithDescriptors.json_decoder)

    @staticmethod
    def json_decoder(obj):
        if "__type__" in obj and obj["__type__"] == LabelsWithDescriptors.CLASS_NAME:
            return LabelsWithDescriptors(obj['index'], obj["labels"],obj["descriptors"])
        return obj

    @staticmethod
    def create_descriptors_from_label_file(f):
        cats = json.load(f)['cats']
        return [LabelsWithDescriptors(i,labels) for (i, labels) in enumerate(cats)]

    class MyEncoder(JSONEncoder):
        def default(self,o):
            d = o.__dict__
            d["__type__"] = LabelsWithDescriptors.CLASS_NAME
            return d


# These are the tests
if __name__ == "__main__":
    if len(argv) != 3:
        print("Usage: python descriptor_test.py -[g|r] [file_path]")
        exit(1)
    elif argv[1] != "-g" and argv[1] != "-r":
        print("Usage: python descriptor_test.py -[g|r] [file_path]")
        exit(2)

    [_, flag, pathR] = argv
    path = os.getcwd() + "/" + pathR
    with open(path) as f:
        if flag == "-g":
            newCats = LabelsWithDescriptors.create_descriptors_from_label_file(f)
            with open(os.getcwd() + "../data/new_cats.json", "w") as outfile:
                json.dump([x for x in newCats], outfile, cls=LabelsWithDescriptors.MyEncoder)
            exit(0)

        lables = LabelsWithDescriptors.read_list_from_file(f)
        for label in lables:
            print(label)
    exit(0)
