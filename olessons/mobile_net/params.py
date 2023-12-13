
#   Notice : Paths here are just used for saving and reading datasets on your own pc, when in colab, use path.yaml to train

train_root_path ='/mnt/d/datasets/petimages/train'
train_tfrecords_path = '/mnt/d/datasets/record/pet_train_224.tfrecords'
train_pkl_path = '../../datasets/train.pkl.gz'
train_npz_path = '../../datasets/train.npz'
train_hdf5_path = '../../datasets/train.hdf5'


val_root_path = '/mnt/d/datasets/petimages/val'
val_tfrecords_path = '../../datasets/pet_val_224.tfrecords'
val_pkl_path = '../../datasets/val.pkl.gz'
val_npz_path = '../../datasets/val.npz'
val_hdf5_path = '../../datasets/val.hdf5'

test_hdf5_path = './test.hdf5'
test_rootpath = '/mnt/d/datasets/petimages/test'





class_yaml_path = './classes.yaml'
yaml_path = './path.yaml'
weights_savelocal_path = './weights'

flip_probability = 0.5
mean,std = [0.485,0.456,0.406],[0.229,0.224,0.225]

batchsize = 50
epochs = 20
learning_rate = 0.0001



