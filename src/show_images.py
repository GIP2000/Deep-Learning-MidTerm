import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from CLIP_runner import get_label_descriptors, build_data, Experiment
import numpy as np



def main():
    lbls = get_label_descriptors(None)
    exp = Experiment(lbls)

    X,Y = build_data(exp,5)

    print("predictions starting")
    (preds, phis) = exp.get_predictionsP(X)
    print("predictions finished")

    for (x,y,pred,phi) in zip(X,Y,preds,phis):
        fig, axis = plt.subplots(2,2)
        axis[0][0].imshow(x)
        axis[0][0].set_title(lbls[y].labels[0])

        axis[1][0].imshow(x)
        axis[1][0].set_title(lbls[y].labels[0])

        descriptors_pred = lbls[pred].descriptors
        descriptor_pred_vals = phi[np.nonzero(exp.descriptor_matrix.T[pred])]
        d_pos_pred = np.arange(len(descriptors_pred))
        axis[0][1].barh(d_pos_pred, descriptor_pred_vals)
        axis[0][1].set_yticks(d_pos_pred,descriptors_pred)
        axis[0][1].invert_yaxis()
        axis[0][1].set_title(f"Prob for Prediction: {lbls[pred].labels[0]}")

        descriptors_true = lbls[y].descriptors
        descriptor_true_vals = phi[np.nonzero(exp.descriptor_matrix.T[y])]
        d_pos_true = np.arange(len(descriptors_true))
        axis[1][1].barh(d_pos_true, descriptor_true_vals)
        axis[1][1].set_yticks(d_pos_true,descriptors_true)
        axis[1][1].invert_yaxis()
        axis[1][1].set_title(f"Prob for correct answer: {lbls[y].labels[0]}")
        plt.show()


if __name__ == '__main__':
    main()
