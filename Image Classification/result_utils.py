import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def plot_curve(history, save_dir):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    min_loss = np.min(val_loss)
    min_epoch = val_loss.index(min_loss)

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.xlim((1, 20))
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.xlim((1, 20))
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(f'{save_dir}/curve.png')
    plt.show()
    print('Epoch= {:.1f}, val_loss= {:.3f}, val_acc= {:.2%}'.format(min_epoch+1, val_loss[min_epoch], val_acc[min_epoch]))

class Result:
    '''
    Write classification report and plot confusion matrix.
    '''
    def __init__(self, test_features, test_labels, class_names, model, save_dir, file_name):
        self.test_features = test_features
        self.test_labels = test_labels
        self.class_names = class_names
        self.save_dir = save_dir
        self.pred_labels = np.argmax(model.predict(test_features), axis=-1)
        self.file_name = file_name

    def write_report(self):
        print(classification_report(
        y_pred=self.pred_labels,
        y_true=self.test_labels,
        target_names=self.class_names,
        output_dict=False))

        report = classification_report(   
        y_pred=self.pred_labels,
        y_true=self.test_labels,
        target_names=self.class_names,
        output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'{self.save_dir}/{self.file_name}.csv')

    def plot_matrix(self, title=None):
        cm = confusion_matrix(y_true=self.test_labels, y_pred=self.pred_labels)
        diagram = sns.heatmap(
            cm, annot=True, fmt='.4g', cmap='Blues',
            xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('True label')
        plt.ylabel('Test Label')
        if title:
            plt.title(title)
        figure = diagram.get_figure()
        figure.savefig(f'{self.save_dir}/{self.file_name}.png')