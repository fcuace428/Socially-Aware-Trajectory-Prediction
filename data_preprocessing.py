from data.data_split import *
from data.mask import *
import os

if __name__ == '__main__':
    if not os.path.isdir('data/titan_data'):
        os.mkdir('data/titan_data')
    if not os.path.isdir('data/titan_data/train'):
        os.mkdir('data/titan_data/train')
        os.mkdir('data/titan_data/val')
        os.mkdir('data/titan_data/test')
    
    # train data
    train_dir = 'data/dataset/Titan/splits/train_set.txt'
    train_list = open(train_dir, "r").read().splitlines()
    train_dir, train_bbox, train_speed, train_intention, train_imu, train_neigh = choice_data(train_list, 'train')
    train_data = np.concatenate((train_bbox, train_speed, train_imu, train_intention), axis=2)
    np.save('data/titan_data/train/train_dir.npy', train_dir)
    np.save('data/titan_data/train/x_train.npy', train_data[:, :10, :])
    np.save('data/titan_data/train/x_train_n.npy', train_neigh[:, :10, :])
    np.save('data/titan_data/train/y_train.npy', train_data[:, 10:, :])
    np.save('data/titan_data/train/y_train_n.npy', train_neigh[:, 10:, :])
    x_train_num = mask_num(train_neigh[:, :10, :])
    y_train_num = mask_num(train_neigh[:, 10:, :])
    np.save('data/titan_data/train/x_train_num1.npy', x_train_num)
    np.save('data/titan_data/train/y_train_num1.npy', y_train_num)
    for_step_wise_y(train_data[:, :10, :], train_data[:, 10:, :], 'train', 'y_train_ts')
    mask_g(train_neigh[:, :10, :], x_train_num, 'train', 'x_mask')
    mask_g(train_neigh[:, 10:, :], y_train_num, 'train', 'y_mask')

    # test data/titan_data
    test_dir = 'data/dataset/Titan/splits/test_set.txt'
    test_list = open(test_dir, "r").read().splitlines()
    test_dir, test_bbox, test_speed, test_intention, test_imu, test_neigh = choice_data(test_list, 'test')
    test_data = np.concatenate((test_bbox, test_speed, test_imu, test_intention), axis=2)
    np.save('data/titan_data/test/test_dir.npy', test_dir)
    np.save('data/titan_data/test/x_test.npy', test_data[:, :10, :])
    np.save('data/titan_data/test/x_test_n.npy', test_neigh[:, :10, :])
    np.save('data/titan_data/test/y_test.npy', test_data[:, 10:, :])
    np.save('data/titan_data/test/y_test_n.npy', test_neigh[:, 10:, :])
    x_test_num = mask_num(test_neigh[:, :10, :])
    y_test_num = mask_num(test_neigh[:, 10:, :])    
    np.save('data/titan_data/test/x_test_num1.npy', x_test_num)
    np.save('data/titan_data/test/y_test_num1.npy', y_test_num)
    for_step_wise_y(test_data[:, :10, :], test_data[:, 10:, :], 'test', 'y_test_ts')
    mask_g(test_neigh[:, :10, :], x_test_num, 'test', 'x_mask')
    mask_g(test_neigh[:, 10:, :], y_test_num, 'test', 'y_mask')

    # val data/titan_data
    val_dir = 'data/dataset/Titan/splits/val_set.txt'
    val_list = open(val_dir, "r").read().splitlines()
    val_dir, val_bbox, val_speed, val_intention, val_imu, val_neigh = choice_data(val_list, 'val')
    val_data = np.concatenate((val_bbox, val_speed, val_imu, val_intention), axis=2)
    np.save('data/titan_data/val/val_dir.npy', val_dir)
    np.save('data/titan_data/val/x_val.npy', val_data[:, :10, :])
    np.save('data/titan_data/val/x_val_n.npy', val_neigh[:, :10, :])
    np.save('data/titan_data/val/y_val.npy', val_data[:, 10:, :])
    np.save('data/titan_data/val/y_val_n.npy', val_neigh[:, 10:, :])
    x_val_num = mask_num(val_neigh[:, :10, :])
    y_val_num = mask_num(val_neigh[:, 10:, :])
    np.save('data/titan_data/val/x_val_num1.npy', x_val_num)
    np.save('data/titan_data/val/y_val_num1.npy', y_val_num)
    for_step_wise_y(val_data[:, :10, :], val_data[:, 10:, :], 'val', 'y_val_ts')
    mask_g(val_neigh[:, :10, :], x_val_num, 'val', 'x_mask')
    mask_g(val_neigh[:, 10:, :], y_val_num, 'val', 'y_mask')
