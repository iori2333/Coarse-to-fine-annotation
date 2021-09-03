import numpy as np
import cv2


def get_confusion_matrix(label, pred, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    seg_pred = pred
    seg_gt = label

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


if __name__ == '__main__':
    thre = 0.7
    label_list = [
        f'F:/spyder_code/ITQ_PyTorch-master/result/val_result_hsi_pca3/0/result{thre}_gray.png',
        f'F:/spyder_code/ITQ_PyTorch-master/result/val_result_hsi_pca3/1/result{thre}_gray.png',
        f'F:/spyder_code/ITQ_PyTorch-master/result/val_result_hsi_pca3/2/result{thre}_gray.png',
        f'F:/spyder_code/ITQ_PyTorch-master/result/val_result_hsi_pca3/3/result{thre}_gray.png',
        f'F:/spyder_code/ITQ_PyTorch-master/result/val_result_hsi_pca3/4/result{thre}_gray.png',
        f'F:/spyder_code/ITQ_PyTorch-master/result/val_result_hsi_pca3/5/result{thre}_gray.png',
    ]
    gt_list = [
        'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180056_110313_josn',
        'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180226_64001_josn',
        'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180241_58359_json',
        'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180535_7407_json',
        'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180641_79313_json',
        'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180919_20370_json',
    ]
    c2f_label = []
    for i in label_list:
        c2f_label.append(cv2.imread(i, cv2.IMREAD_GRAYSCALE))

    gt = []
    h, w = 1773, 1379
    for j in gt_list:
        gt.append(cv2.resize(cv2.imread(j + '/label_gray.png', cv2.IMREAD_GRAYSCALE).transpose(1, 0)[:, ::-1],
                             (int(w / 2), int(h / 2)),
                             interpolation=cv2.INTER_NEAREST)
                  )

    num_class = 10
    confusion_matrix = np.zeros((num_class, num_class))

    for num in range(len(label_list)):
        confusion_matrix += get_confusion_matrix(gt[num], c2f_label[num],
                                                 num_class=num_class, ignore=-1)
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    acc_array = tp / np.maximum(1.0, pos)
    mean_acc = acc_array[1:].mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array[1:].mean()
    print(f'MIou: {IoU_array}')
    print(mean_IoU)
    print(f'acc: {acc_array}')
    print(mean_acc)
