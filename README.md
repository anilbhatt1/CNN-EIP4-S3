# CNN-EIP4-S3
Session 3 Assignments
Final Validation accuracy for Base Network
------------------------------------------
Accuracy on test data is: 82.55
Your model definition (model.add... ) with output channel size and receptive field
-----------------------------------------------------------------------------------
# Define the model as model_new
model_new= Sequential()
model_new.add(SeparableConv2D(filters = 32, kernel_size =(3, 3), strides=1, padding='valid', dilation_rate= 1, 
                              activation = 'relu', input_shape=(32, 32, 3))) # 30x30x32, RF 3
model_new.add(BatchNormalization())
model_new.add(Dropout(0.1))
model_new.add(SeparableConv2D(filters = 64, kernel_size =(3, 3), strides=1, padding='valid', dilation_rate= 1, 
                              activation = 'relu')) # 28x28x64, RF 5
model_new.add(BatchNormalization())
model_new.add(Dropout(0.1))
model_new.add(MaxPooling2D(pool_size=(2,2))) # 14x14x64, RF 6
model_new.add(SeparableConv2D(filters = 64, kernel_size =(3, 3), strides=1, padding='valid', dilation_rate= 1, 
                              activation = 'relu')) # 12x12x64, RF 10 
model_new.add(BatchNormalization())
model_new.add(Dropout(0.1))
model_new.add(SeparableConv2D(filters = 128, kernel_size =(3, 3), strides=1, padding='valid', dilation_rate= 1, 
                              activation = 'relu')) # 10x10x128, RF 14
model_new.add(BatchNormalization())
model_new.add(Dropout(0.1))
model_new.add(SeparableConv2D(filters = 256, kernel_size =(3, 3), strides=1, padding='valid', dilation_rate= 1, 
                              activation = 'relu')) # 8x8x256, RF 18
model_new.add(BatchNormalization())
model_new.add(Dropout(0.1))
model_new.add(SeparableConv2D(filters = 64, kernel_size =(3, 3), strides=1, padding='valid', dilation_rate= 1, 
                              activation = 'relu')) # 6x6x64, RF 22
model_new.add(BatchNormalization())
model_new.add(Dropout(0.1))
model_new.add(SeparableConv2D(filters = 32, kernel_size =(3, 3), strides=1, padding='valid', dilation_rate= 1, 
                              activation = 'relu')) # 4x4x32, RF 32
model_new.add(BatchNormalization())
model_new.add(Dropout(0.1))
model_new.add(SeparableConv2D(filters = 10, kernel_size =(4, 4), strides=1, padding='valid', dilation_rate= 1, 
                              activation = 'relu')) # 1x1x10, RF 32
model_new.add(Flatten())             # 10,
model_new.add(Activation('softmax')) # 10 equal to number of classes
# Compile the model
model_new.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

Your 50 epoch logs - Max test accuracy received 0.7977
------------------
Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.004.
  2/781 [..............................] - ETA: 1:08 - loss: 0.6982 - acc: 0.7734/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:16: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  app.launch_new_instance()
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:16: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., verbose=1, callbacks=[<keras.ca..., steps_per_epoch=781, epochs=50)`
  app.launch_new_instance()
781/781 [==============================] - 44s 56ms/step - loss: 0.7843 - acc: 0.7255 - val_loss: 0.8275 - val_acc: 0.7209
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0030326005.
781/781 [==============================] - 43s 55ms/step - loss: 0.7384 - acc: 0.7401 - val_loss: 0.8208 - val_acc: 0.7288
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0024420024.
781/781 [==============================] - 43s 55ms/step - loss: 0.7131 - acc: 0.7531 - val_loss: 0.7722 - val_acc: 0.7455
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0020439448.
781/781 [==============================] - 43s 55ms/step - loss: 0.6877 - acc: 0.7589 - val_loss: 0.7215 - val_acc: 0.7585
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0017574692.
781/781 [==============================] - 43s 55ms/step - loss: 0.6703 - acc: 0.7671 - val_loss: 0.7022 - val_acc: 0.7632
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0015414258.
781/781 [==============================] - 43s 55ms/step - loss: 0.6480 - acc: 0.7724 - val_loss: 0.7222 - val_acc: 0.7586
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0013726836.
781/781 [==============================] - 43s 55ms/step - loss: 0.6282 - acc: 0.7793 - val_loss: 0.7270 - val_acc: 0.7590
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.001237241.
781/781 [==============================] - 43s 55ms/step - loss: 0.6166 - acc: 0.7838 - val_loss: 0.6960 - val_acc: 0.7676
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0011261261.
781/781 [==============================] - 43s 55ms/step - loss: 0.6104 - acc: 0.7841 - val_loss: 0.7435 - val_acc: 0.7569
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0010333247.
781/781 [==============================] - 43s 55ms/step - loss: 0.5934 - acc: 0.7921 - val_loss: 0.6913 - val_acc: 0.7711
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0009546539.
781/781 [==============================] - 43s 55ms/step - loss: 0.5896 - acc: 0.7948 - val_loss: 0.6552 - val_acc: 0.7822
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0008871147.
781/781 [==============================] - 43s 55ms/step - loss: 0.5782 - acc: 0.7973 - val_loss: 0.6703 - val_acc: 0.7768
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0008285004.
781/781 [==============================] - 43s 55ms/step - loss: 0.5733 - acc: 0.7975 - val_loss: 0.6623 - val_acc: 0.7843
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0007771517.
781/781 [==============================] - 43s 54ms/step - loss: 0.5678 - acc: 0.8007 - val_loss: 0.6432 - val_acc: 0.7900
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0007317966.
781/781 [==============================] - 43s 56ms/step - loss: 0.5632 - acc: 0.8007 - val_loss: 0.6360 - val_acc: 0.7881
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0006914434.
781/781 [==============================] - 45s 57ms/step - loss: 0.5592 - acc: 0.8035 - val_loss: 0.6424 - val_acc: 0.7894
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.000655308.
781/781 [==============================] - 44s 56ms/step - loss: 0.5467 - acc: 0.8087 - val_loss: 0.6454 - val_acc: 0.7865
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0006227619.
781/781 [==============================] - 43s 56ms/step - loss: 0.5459 - acc: 0.8085 - val_loss: 0.6396 - val_acc: 0.7890
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0005932958.
781/781 [==============================] - 44s 56ms/step - loss: 0.5428 - acc: 0.8104 - val_loss: 0.6495 - val_acc: 0.7873
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.000566492.
781/781 [==============================] - 44s 56ms/step - loss: 0.5337 - acc: 0.8130 - val_loss: 0.6638 - val_acc: 0.7834
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0005420054.
781/781 [==============================] - 45s 57ms/step - loss: 0.5361 - acc: 0.8116 - val_loss: 0.6472 - val_acc: 0.7913
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.000519548.
781/781 [==============================] - 45s 58ms/step - loss: 0.5359 - acc: 0.8120 - val_loss: 0.6462 - val_acc: 0.7876
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0004988775.
781/781 [==============================] - 45s 57ms/step - loss: 0.5257 - acc: 0.8141 - val_loss: 0.6623 - val_acc: 0.7835
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0004797889.
781/781 [==============================] - 45s 58ms/step - loss: 0.5272 - acc: 0.8177 - val_loss: 0.6344 - val_acc: 0.7947
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0004621072.
781/781 [==============================] - 45s 58ms/step - loss: 0.5218 - acc: 0.8151 - val_loss: 0.6441 - val_acc: 0.7914
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0004456825.
781/781 [==============================] - 45s 58ms/step - loss: 0.5273 - acc: 0.8148 - val_loss: 0.6421 - val_acc: 0.7915
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0004303852.
781/781 [==============================] - 45s 58ms/step - loss: 0.5182 - acc: 0.8173 - val_loss: 0.6474 - val_acc: 0.7870
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0004161032.
781/781 [==============================] - 45s 58ms/step - loss: 0.5179 - acc: 0.8195 - val_loss: 0.6291 - val_acc: 0.7964
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0004027386.
781/781 [==============================] - 45s 58ms/step - loss: 0.5114 - acc: 0.8206 - val_loss: 0.6338 - val_acc: 0.7947
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0003902058.
781/781 [==============================] - 45s 57ms/step - loss: 0.5063 - acc: 0.8226 - val_loss: 0.6393 - val_acc: 0.7914
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0003784295.
781/781 [==============================] - 45s 57ms/step - loss: 0.5154 - acc: 0.8182 - val_loss: 0.6311 - val_acc: 0.7952
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0003673432.
781/781 [==============================] - 45s 57ms/step - loss: 0.5088 - acc: 0.8200 - val_loss: 0.6302 - val_acc: 0.7971
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0003568879.
781/781 [==============================] - 45s 58ms/step - loss: 0.5092 - acc: 0.8210 - val_loss: 0.6312 - val_acc: 0.7977
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0003470114.
781/781 [==============================] - 45s 58ms/step - loss: 0.5011 - acc: 0.8230 - val_loss: 0.6344 - val_acc: 0.7958
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0003376667.
781/781 [==============================] - 45s 58ms/step - loss: 0.5049 - acc: 0.8212 - val_loss: 0.6303 - val_acc: 0.7948
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0003288122.
781/781 [==============================] - 45s 57ms/step - loss: 0.5010 - acc: 0.8240 - val_loss: 0.6371 - val_acc: 0.7928
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0003204101.
781/781 [==============================] - 44s 57ms/step - loss: 0.5017 - acc: 0.8233 - val_loss: 0.6481 - val_acc: 0.7905
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0003124268.
781/781 [==============================] - 45s 57ms/step - loss: 0.4975 - acc: 0.8244 - val_loss: 0.6284 - val_acc: 0.7941
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0003048316.
781/781 [==============================] - 45s 58ms/step - loss: 0.4957 - acc: 0.8273 - val_loss: 0.6301 - val_acc: 0.7952
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0002975969.
781/781 [==============================] - 46s 59ms/step - loss: 0.4956 - acc: 0.8272 - val_loss: 0.6470 - val_acc: 0.7921
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0002906977.
781/781 [==============================] - 45s 57ms/step - loss: 0.4926 - acc: 0.8264 - val_loss: 0.6449 - val_acc: 0.7920
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0002841111.
781/781 [==============================] - 45s 57ms/step - loss: 0.4954 - acc: 0.8262 - val_loss: 0.6263 - val_acc: 0.7957
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0002778164.
781/781 [==============================] - 45s 58ms/step - loss: 0.4943 - acc: 0.8267 - val_loss: 0.6289 - val_acc: 0.7974
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0002717945.
781/781 [==============================] - 45s 58ms/step - loss: 0.4810 - acc: 0.8294 - val_loss: 0.6543 - val_acc: 0.7906
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0002660282.
781/781 [==============================] - 46s 58ms/step - loss: 0.4854 - acc: 0.8280 - val_loss: 0.6485 - val_acc: 0.7900
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0002605015.
781/781 [==============================] - 45s 58ms/step - loss: 0.4853 - acc: 0.8287 - val_loss: 0.6447 - val_acc: 0.7922
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0002551997.
781/781 [==============================] - 45s 58ms/step - loss: 0.4815 - acc: 0.8302 - val_loss: 0.6436 - val_acc: 0.7927
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0002501094.
781/781 [==============================] - 45s 58ms/step - loss: 0.4865 - acc: 0.8289 - val_loss: 0.6376 - val_acc: 0.7952
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0002452182.
781/781 [==============================] - 45s 58ms/step - loss: 0.4863 - acc: 0.8290 - val_loss: 0.6398 - val_acc: 0.7936
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0002405147.
781/781 [==============================] - 45s 58ms/step - loss: 0.4861 - acc: 0.8297 - val_loss: 0.6371 - val_acc: 0.7932
Model_new took 2216.39 seconds to train

Accuracy on test data is: 79.32
