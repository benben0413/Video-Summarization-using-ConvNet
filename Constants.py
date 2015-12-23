from theano.tensor import tanh
mini_batch_size=1
filter_shape1=(10,1,7,7)
filter_shape2=(6,2,7,7)
poolsize=(2,2)
activation_fn=tanh
image_shape1=(1,3,64,64)
image_shape2=(1,3,128,128)
image_shape3=(1,3,256,256)
num_class=33
image_size=256
train_size=1800
num_training_batches=36
num_validation_batches=18
random_num_filters=[64,256]
net_out_size=[29,51,125]

