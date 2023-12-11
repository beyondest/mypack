import tensorflow as tf
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch


class tfrecord():
    def __init__(self) -> None:
        pass
    
    
    @classmethod
    def save_to_tfrecords(cls,
                          dataset:Dataset,
                          records_save_path:str,
                          save_X_type:type=float,
                          save_y_type:type=float):
        """Save pytorch dataset to tfrecords type file
        Use 'X' and 'y' in feature_dict
        Args:
            dataset (Dataset): _description_
            records_save_path (str): _description_
            save_X_type (type, int | float | bytes ): _description_. Defaults to float.
            save_y_type (type, int | float | bytes ): _description_. Defaults to float.
        """
        with tf.io.TFRecordWriter(records_save_path) as writer:
            for X,y in dataset:
                numpy_X = np.asanyarray(X)
                numpy_y = np.asanyarray(y)
                if save_X_type == float:
                    
                    X_feature = tf.train.Feature(float_list = tf.train.FloatList(value = numpy_X.flatten()))
                elif save_X_type == int:
                    X_feature = tf.train.Feature(int64_list = tf.train.Int64List(value = numpy_X.flatten()))
                elif save_X_type == bytes:
                    X_feature = tf.train.Feature(bytes_list = tf.train.BytesList(value = numpy_X.flatten()))
                if save_y_type == float:
                    
                    y_feature = tf.train.Feature(float_list = tf.train.FloatList(value = numpy_y.flatten()))
                elif save_y_type == int:
                    y_feature = tf.train.Feature(int64_list = tf.train.Int64List(value = numpy_y.flatten()))
                elif save_y_type == bytes:
                    y_feature = tf.train.Feature(bytes_list = tf.train.BytesList(value = numpy_y.flatten()))
                      
                feature = tf.train.Features(feature={'X':X_feature,'y':y_feature})
                example = tf.train.Example(features=feature)
                
                writer.write(example.SerializeToString())
                print(f'tfrecords save to {records_save_path}')
                print(f'X shape is {numpy_X.shape}, X dtype is {save_X_type}')
                print(f'y shape is {numpy_y.shape}, y dtype is {save_y_type}')
                

    
    
    @classmethod
    def show_feature_list(cls,tfrecord_path:str):
        """Use this to show feature list of tfrecords if you do not know what in it

        Returns:
            _type_: _description_
        """
        
        tf_dataset = tf.data.TFRecordDataset(tfrecord_path)
        for raw_record in tf_dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
        print(f'Features In {tfrecord_path}:') 
        print(list(example.features.feature.keys()))
        

    class olddatasets(Dataset):
        def __init__(self,tfrecord_path:str,transforms) -> None:
            """You have to change feature_description each time!!!

            Args:
                tfrecord_path (str): _description_
                transforms (_type_): _description_
                version (int, optional): _description_. Defaults to 1.

            Raises:
                TypeError: _description_
            """
            super().__init__()
            self.tfrecord_path = tfrecord_path
            self.tf_dataset = tf.data.TFRecordDataset(tfrecord_path)
            self.len = len(list(self.tf_dataset))
            self.transforms = transforms

            self.feature_description  = {
                'height':tf.io.FixedLenFeature([],tf.int64),
                'width': tf.io.FixedLenFeature([], tf.int64),
                'depth': tf.io.FixedLenFeature([], tf.int64),
                'label': tf.io.FixedLenFeature([], tf.int64),
                'image_raw': tf.io.FixedLenFeature([], tf.string),
            }
            self.parsed = self.tf_dataset.map(self.parse_tfrecord_fn)
        def __len__(self):
            return self.len

        def parse_tfrecord_fn(self, record):
            
            example = tf.io.parse_single_example(record, self.feature_description)
            width = example['width']
            depth = example['depth']
            height = example['height']
            label = example['label']
            image_raw = example['image_raw']
            

                
            
            return {'width':width,'depth':depth,'height':height,'label':label,'image_raw':image_raw}

        def __getitem__(self, idx):
            sample = next(iter(self.parsed.take(idx + 1).skip(idx)))
            width = sample['width']
            depth = sample['depth']
            height = sample['height']
            label = sample['label']
            image_raw = sample['image_raw']
            
            image = tf.compat.v1.decode_raw(image_raw,tf.uint8)
            image = tf.reshape(image,(height,width,depth))
            
            # Warning : X,y must be np.array and dtype must be np.int before transform!!!!
            X,y = np.asanyarray(image),np.asanyarray(label)
            X = self.transforms(X)
            y = torch.tensor(y)
            
            return X,y
    
    class datasets(Dataset):
        def __init__(self,tfrecord_path:str,
                     transforms,
                     X_shape:list,
                     y_shape:list =[1],
                     X_type:type = float,
                     y_type:type = float) -> None:
            """Only suppport for tfrecords saved by my custom tfrecord class
            notice:X_shape should be [hei,wid,channels]
            Args:
                tfrecord_path (str): _description_
                transforms (_type_): _description_

            Raises:
                TypeError: _description_
            """
            super().__init__()
            self.tfrecord_path = tfrecord_path
            self.tf_dataset = tf.data.TFRecordDataset(tfrecord_path)
            self.len = len(list(self.tf_dataset))
            self.transforms = transforms
            
            if X_type == int:
                X_feature = tf.io.FixedLenFeature([*X_shape],tf.int64)
            elif X_type == float:
                X_feature = tf.io.FixedLenFeature([*X_shape],tf.float32)
            elif X_type == bytes:
                X_feature = tf.io.FixedLenFeature([*X_shape],tf.string)
            else:
                raise TypeError('Wrong type')
            if y_type == int:
                y_feature = tf.io.FixedLenFeature([*y_shape],tf.int64)
            elif y_type == float:
                y_feature = tf.io.FixedLenFeature([*y_shape],tf.float32)
            elif y_type == bytes:
                y_feature = tf.io.FixedLenFeature([*y_shape],tf.string)
            else:
                raise TypeError('Wrong type')
            self.feature_description  = {
                'X':X_feature,
                'y':y_feature 
            }
            self.parsed_dataset = self.tf_dataset.map(self._parse_tfrecord_fn)


            
        def __len__(self):
            return len(list(self.parsed_dataset))

        def __getitem__(self, idx):
            sample = next(iter(self.parsed_dataset.take(idx + 1).skip(idx)))
            X = np.asanyarray(sample['X'])
            y = np.asanyarray(sample['y'])
            X = self.transforms(X)
            return X, y

        def _parse_tfrecord_fn(self, example_proto):
            example = tf.io.parse_single_example(example_proto, self.feature_description)
            return {'X': example['X'], 'y': example['y']}
