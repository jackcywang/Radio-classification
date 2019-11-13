# -*- coding: utf-8 -*-
from keras import layers
from keras.layers import Input
from keras.layers.core import Activation,Flatten,Dense, Reshape
from keras.layers import Conv1D
from keras.layers import BatchNormalization
from keras.models import Model
def identity_block(input_tensor, kernel_size, filters):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
    # Returns
        Output tensor for the block.
    """
    filters1, filters2,filters3 = filters
    kernel_size1,kernel_size2,kernel_size3 = kernel_size


    x = Conv1D(filters1, kernel_size1,padding='same' )(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters2, kernel_size2,padding='same' )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters3, kernel_size3,padding='same')(x)
    x = BatchNormalization()(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters , strides=(2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2,filters3 = filters
    kernel_size1,kernel_size2,kernel_size3 = kernel_size

#    x = ZeroPadding1D((2))(input_tensor)
    x = Conv1D(filters1, kernel_size1, strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters2, kernel_size2,padding='same' )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters3, kernel_size3, padding='same' )(x)
    x = BatchNormalization()(x)

    shortcut = Conv1D(filters3, kernel_size1, strides=strides)(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(in_shp,classes,include_top=True, 
             input_tensor=None, 
             pooling=None,input_shape=None
             ):
    
    # Determine proper input shape

#    if input_tensor is None:
#    input_shape=Reshape((in_shp+[1]), input_shape=in_shp)
#    input_shape = _obtain_input_shape(input_shape)
#    input_shape = _obtain_input_shape(input_shape,
#                                      default_size=128,
#                                      min_size=1,
#                                      data_format=K.image_data_format(),
#                                      include_top=True)
    img_input = Input(shape=(512,1),batch_shape=None)
#    img_input=Reshape((128,1),)
#    else:
#        if not K.is_keras_tensor(input_tensor):
#            img_input = Input(tensor=input_tensor, shape=input_shape)
#        else:
#            img_input = input_tensor

#    x = ZeroPadding1D((3))(img_input)
    x = Conv1D(16, 17, padding='same', activation="relu",name='conv1')(img_input)
    x = BatchNormalization()(x)
    x = Conv1D(32, 19, padding='same', activation="relu",name='conv2')(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, 21, padding='same', activation="relu",name='conv3')(x)
    x = BatchNormalization()(x)
##    x = Activation('relu')(x)
#    x = MaxPooling1D(pool_size=2, strides=None, padding='same')(x)

    x = conv_block(x, [15,17,19],[32,32, 64])
    x = identity_block(x, [15,17,19], [32,32, 64])
    
    x = conv_block(x, [17,19,21],[32,32, 64])
    x = identity_block(x, [17,19,21], [32,32, 64])
    

#    x = AveragePooling1D((7), name='avg_pool',padding='same')(x)

    x=Flatten()(x)
    x=Dense(64, activation='relu', init='he_normal', name="dense1")(x)
    x=BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(x)
#    x=Dropout(0.2)(x)
    x=Dense(classes, init='he_normal', name="dense2")(x)
    x=BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(x)
    x=Activation('softmax')(x)
    x=Reshape([classes])(x)

#    if include_top:
#        x = Flatten()(x)
#        x = Dense(classes, activation='softmax', name='fc1000')(x)
#    else:
#        if pooling == 'avg':
#            x = GlobalAveragePooling1D()(x)
#        elif pooling == 'max':
#            x = GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
#    if input_tensor is not None:
#        inputs = get_source_inputs(input_tensor)
#    else:
    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')
#    model.load_weights('convmodrecnets_resnet9_31.wts.h5', by_name=True)
    return model