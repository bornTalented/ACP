import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

def plot_acc_loss(history, accuracyName, lossName, savePDF=False, ModelName='Model'):
    print(history.history.keys())

    # dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

    def plot_curve(curveName, legendPos, savePDF):

        plt.plot(history.history[curveName])

        if 'val_' + curveName in history.history.keys():
            plt.plot(history.history['val_' + curveName])

        plt.title(ModelName)
        plt.ylabel(curveName)
        plt.xlabel('Epoch')
        #     plt.xticks(np.arange(0, 110, step=10))
        plt.legend(['Train', 'Val'], loc=legendPos)

        plt.xticks(range(len(history.history[curveName])), range(1, len(history.history[curveName]) + 1))
        
        if savePDF:
            plt.savefig(ModelName + "_" + curveName + ".pdf", format="pdf")

        # plt.show()

    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 5))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    plot_curve(accuracyName, 'lower right', savePDF)
    
    plt.subplot(1, 2, 2) # row 1, col 2 index 2
    plot_curve(lossName, 'upper right', savePDF)
    
    plt.show()


def plot_seaborn_confusion_matrix(y_true, y_pred, class_labels, normalize=None, imageName=False):
    from sklearn.metrics import confusion_matrix

    data = confusion_matrix(y_true, y_pred,
                            normalize=normalize)  # normalize must be one of {'true', 'pred', 'all', None}
    df_cm = pd.DataFrame(data, columns=class_labels, index=class_labels)
    # df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
    df_cm.index.name = '_________________Actual_________________'
    df_cm.columns.name = '________________Predicted________________'
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)  # for label size

    if normalize:
        fmt = '.2g'
    else:
        fmt = 'g'

    svm = sn.heatmap(df_cm, cmap="Blues", annot=True, fmt=fmt, annot_kws={"size": 20}, square=True, vmax=500, # vmax only of ACSP results
                     linewidths=.2)  # font size

    # print(np.count_nonzero(y_true == 1), np.count_nonzero(y_pred == 1))

    if imageName:
        # imageName with extension e.g. "ConfusionMatrix.pdf" or "ConfusionMatrix.png"
        Name = imageName.split('.')
        figure = svm.get_figure()
        if Name[-1] == 'pdf':
            figure.savefig(imageName, format="pdf")
        else:
            figure.savefig(imageName, dpi=400)
