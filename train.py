from nets.deeplab import Deeplabv3
from keras.utils.data_utils import get_file
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
import keras
from keras import backend as K
import numpy as np

ALPHA = 1.0
WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
                    
NCLASSES = 2
HEIGHT = 416
WIDTH = 416

def generate_arrays_from_file(lines,batch_size):
    #print(lines)
    # 获取总长度
    n = len(lines)
    #print(n)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            #print(i)
            #print(lines[i])
            name = lines[i].replace("\n", "") # name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = Image.open(r"dataset2/jpg" + '/' + name + ".png")
            img = img.resize((WIDTH,HEIGHT),Image.BICUBIC)
            img = np.array(img)
            img = img/255
            X_train.append(img)

            name = lines[i].replace("\n", "") # name = (lines[i].split(';')[1]).replace("\n", "")
            # 从文件中读取图像
            img = Image.open(r"dataset2/png" + '/' + name + ".png")
            img = img.resize((int(WIDTH),int(HEIGHT)),Image.NEAREST)
            # 更改
            img = np.array(img.convert('RGB'))# img = np.array(img)
            
            seg_labels = np.zeros((int(HEIGHT),int(WIDTH),NCLASSES))
            
            # 更改
            #for c in range(NCLASSES):
                #seg_labels[: , : , c ] = (img[:,:,0] == c ).astype(int)
#             if np.all(img[:,:,0] == img[0,0,0]):
#                 # black-only
#                 for c in range(NCLASSES):
#                     seg_labels[:,:,c] = 1 if img[0,0,0] == c else 0
#             else:
#                 for c in range(NCLASSES):
#                     seg_labels[:, :, c] = (img[:, :, 0] == c).astype(int)
            
#             #seg_labels = np.reshape(seg_labels, (-1,NCLASSES))
#             label = np.reshape(np.array(seg_labels), [-1])
#             one_hot_label = np.eye(NCLASSES)[np.array(label, np.int32)]
#             Y_train.append(one_hot_label)

#             # 读完一个周期后重新开始
#             i = (i+1) % n

            if np.all(img[:,:,0] == img[0,0,0]):
                # black-only
                #for c in range(NCLASSES):
                    #seg_labels[:,:,c] = 1    # 1 for black
                seg_labels[:,:,0] = 1
                seg_labels[:,:,1] = 0    # white
            else:
                #for c in range(NCLASSES):
                    #seg_labels[:, :, c] = (img[:, :, 0] == 255).astype(int)
                for c in range(NCLASSES):
                    seg_labels[: , : , c ] = (img[:,:,0] == c*255 ).astype(int)
            
            seg_labels = np.reshape(seg_labels, (-1,NCLASSES))
            Y_train.append(seg_labels)

            # 读完一个周期后重新开始
            i = (i+1) % n
        yield (np.array(X_train),np.array(Y_train))

def loss(y_true, y_pred):
    crossloss = K.binary_crossentropy(y_true,y_pred)
    loss = K.sum(crossloss)/HEIGHT/WIDTH
    return loss

if __name__ == "__main__":
    log_dir = "logs/"
    # 获取model
    model = Deeplabv3(classes=2,input_shape=(HEIGHT,WIDTH,3))
    # model.summary()

    #weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
    #                                WEIGHTS_PATH_X,
    #                                cache_subdir='models')
    weights_path = "models/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
    model.load_weights(weights_path,by_name=True)

    # 打开数据集的txt
    with open(r"dataset2/train.txt","r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.125)
    num_train = len(lines) - num_val

    # 保存的方式，1世代保存一次
    checkpoint_period = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', 
                                    save_weights_only=True, 
                                    save_best_only=False, 
                                    period=1
                                )
    # 学习率下降的方式，val_loss 1次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss', 
                            factor=0.5, 
                            patience=3, 
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            min_delta=0, 
                            patience=10, 
                            verbose=1
                        )

    # 交叉熵
    model.compile(loss = loss,
            optimizer = Adam(lr=1e-3),
            metrics = ['accuracy'])
            
    batch_size = 1
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=3,
            initial_epoch=0,
            callbacks=[checkpoint_period, reduce_lr])

    model.save_weights(log_dir+'last1.h5')
