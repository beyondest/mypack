

train_root_path ='/mnt/d/datasets/petimages/train'
train_tfrecords_path = '/mnt/d/datasets/record/pet_train_224.tfrecords'
train_pkl_path = './train.pkl.gz'


val_root_path = '/mnt/d/datasets/petimages/val'
val_tfrecords_path = '/mnt/d/datasets/record/pet_val_224.tfrecords'
val_pkl_path = '../../datasets/val.pkl.gz'

test_root_path = '/mnt/d/datasets/petimages/test'
test_tfrecords_path ='.'
test_pkl_path = './test.pkl.gz'

weights_savepath = './weights.pth'


class_yaml_path = '../../datasets/classes.yaml'
mean,std = [0.485,0.456,0.406],[0.229,0.224,0.225]

batchsize = 20
epochs = 10
learning_rate = 0.0001



