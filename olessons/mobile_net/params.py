
#   Notice : Paths here are just used for saving and reading datasets on your own pc, when in colab, use path.yaml to train

train_root_path ='/mnt/d/datasets/petimages/train'
train_tfrecords_path = '/mnt/d/datasets/record/pet_train_224.tfrecords'
train_pkl_path = '../../datasets/train.pkl.gz'
train_npz_path = '../../datasets/train.npz'

val_root_path = '/mnt/d/datasets/petimages/val'
val_tfrecords_path = '/mnt/d/datasets/record/pet_val_224.tfrecords'
val_pkl_path = '../../datasets/val.pkl.gz'
train_npz_path = '../../datasets/val.npz'









class_yaml_path = './classes.yaml'
yaml_path = './path.yaml'

mean,std = [0.485,0.456,0.406],[0.229,0.224,0.225]

batchsize = 200
epochs = 10
learning_rate = 0.0001



