import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from CLIP_runner import get_label_descriptors, build_data, Experiment
import numpy as np


def make_image(axis, title: str, img):
    axis.imshow(img)
    axis.set_title(title)
    return axis


def main():
    lbls = get_label_descriptors(None)
    exp = Experiment(lbls)

    X,Y = build_data(exp,50)

    print("Our predictions starting")
    (preds_o, phis_o) = exp.get_predictionsDP(X)

    print("CLIP predictions starting")
    (preds_c, phis_c) = exp.get_predictionsOP(X)
    print("Predictions finished")

    for (x,y,pred_o,phi_o,pred_c,phi_c) in zip(X,Y,preds_o,phis_o,preds_c,phis_c):
        fig, axis = plt.subplots(3,2, layout="constrained")
        axis[0][0] = make_image(axis[0][0], f"Ours: {lbls[pred_o].labels[0]}", x)

        axis[1][0] = make_image(axis[1][0], f"Answer: {lbls[y].labels[0]}", x)

        axis[2][0] = make_image(axis[2][0], f"CLIPs: {lbls[pred_c].labels[0]}", x)

        descriptors_pred = lbls[pred_o].descriptors
        descriptor_pred_vals = phi_o[np.nonzero(exp.descriptor_matrix.T[pred_o])]
        d_pos_pred = np.arange(len(descriptors_pred))
        axis[0][1].barh(d_pos_pred, descriptor_pred_vals)
        axis[0][1].set_yticks(d_pos_pred,descriptors_pred)
        axis[0][1].invert_yaxis()
        axis[0][1].set_title("Probs")

        descriptors_true = lbls[y].descriptors
        descriptor_true_vals = phi_o[np.nonzero(exp.descriptor_matrix.T[y])]
        d_pos_true = np.arange(len(descriptors_true))
        axis[1][1].barh(d_pos_true, descriptor_true_vals)
        axis[1][1].set_yticks(d_pos_true,descriptors_true)
        axis[1][1].invert_yaxis()
        axis[1][1].set_title("Probs")

        descriptors_CLIP = lbls[pred_c].descriptors
        descriptor_CLIP_vals = phi_o[np.nonzero(exp.descriptor_matrix.T[pred_c])]
        d_pos_CLIP = np.arange(len(descriptors_CLIP))
        axis[2][1].barh(d_pos_CLIP, descriptor_CLIP_vals)
        axis[2][1].set_yticks(d_pos_CLIP,descriptors_CLIP)
        axis[2][1].invert_yaxis()
        axis[2][1].set_title("Probs")

        plt.savefig(f"{'_'.join(lbls[y].labels[0].split(' '))}.png", dpi=300)


if __name__ == '__main__':
    main()
