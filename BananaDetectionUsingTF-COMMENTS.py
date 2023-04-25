import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

def build_model(input_shape):
    inputs = L.Input(input_shape)

    #backbone
    #The MobileNetV2 implementation in tensorflow.keras.applications has 155 layers in total
    #but len(backkbone.layers) = 154, since we have not included the top classification layer by setting `include_top=False`
    #All the explanation about the model in #BananaDetectionUsingTF onenote page
    backbone = MobileNetV2(
            include_top = False,
            weights = "imagenet",
            input_tensor = inputs,
            alpha=1.0
        )
    print('the number of layers in backbone', len(backbone.layers))
    
    #Detection Head
    #x = backbone.get_layer("block_13_expand_relu").output
    x = backbone.output
    print('first shape',x.shape)
    #decreasing the number of channels
    #adding convolutional layer
    x = L.Conv2D(256, kernel_size = 1, padding = "same")(x)
    print('second shape',x.shape)
    #adding batch normalization layer
    x = L.BatchNormalization()(x)
    print('third shape',x.shape)
    #adding rectified linear unit (ReLU) activation function
    x = L.Activation("relu")(x)
    print('fourth shape',x.shape)
    #adding global pooling average layer
    x = L.GlobalAveragePooling2D()(x)
    print('fifth shape',x.shape)
    #adding dropout layer
    x = L.Dropout(0.5)(x)
    print('sixth shape',x.shape)
    #adding a fully conneected dense layer with 4 units and a sigmoid activation function, which produces the final output
    x = L.Dense(4, activation="sigmoid")(x)
    #print('seventh shape',x.shape)
    #print(x.shape)
    
    #Model
    model = Model(inputs,x)
    return model

if __name__ == "__main__":
    input_shape = (256,256,3)
    build_model(input_shape)
    model = build_model(input_shape)
    #print('the number of layers in model', len(model.layers))
    #model.summary()