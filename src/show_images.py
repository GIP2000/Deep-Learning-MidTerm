import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from CLIP_runner import get_label_descriptors, build_data, Experiment



def main():
    lbls = get_label_descriptors(5)
    exp = Experiment(lbls)


    X,Y = build_data(exp,5)

    for (x,y) in zip(X,Y):
        image = mpimg.imread(x)
        plt.imshow(image)
        plt.title(lbls[y].labels[0])
        plt.show()


if __name__ == '__main__':
    main()
