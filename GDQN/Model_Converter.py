import tensorflow as tf



class GeneticModel:
    def __init__(self, Mgen, Sgen,optim=tf.keras.optimizers.Adam,lr=1e-7):
        self.lr = lr; self.optim = optim(learning_rate=lr)
        self.model = self.Convert2Model(Mgen,Sgen)
        pass


    def Calc_Conv_HP(self, PS_gen1, PS_gen2):
        Kernel_Size = PS_gen1 - PS_gen2 +1
        return Kernel_Size

    def Calc_Pool_HP(self, PS_gen1, PS_gen2):
        strides = 1; pooling_size = 1
        if((PS_gen1+1)%(PS_gen2+1)==0):
            strides = int((PS_gen1+1)/(PS_gen2+1)); pooling_size = strides
        else:
            strides = int((PS_gen1+2)/(PS_gen2+1)); pooling_size = strides-1
        
        return strides, pooling_size


    def Calc_ConvTranspose_HP(self,PS_gen1, PS_gen2):
        kernel_size = PS_gen2-PS_gen1+1
        return kernel_size


    def Calc_UpSampling_HP(self, PS_gen1, PS_gen2):
        size = int(PS_gen2/PS_gen1)
        return size
    
    
    def Convert2Model(self,Mgen,Sgen):
        print('Converting')
        Mgen = (lambda x: [x[3 * i:3 * i + 3] for i in range(11)])(Mgen)
        Sgen = (lambda x: [x[2 * i:2 * i + 2] for i in range(11)])(Sgen)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(1,input_shape=(84,84,4)))
        for i in range(1):
            Prev = 84
            j = 0
            while(((int(Sgen[j],16)==255)or(int(Sgen[j],16)==0))==1):
                j+=1
            Post = int(Sgen[j],16)

            print(Prev,Post)
            if (Prev >= Post):
                if (Mgen[i] == '010'):
                    # Sgen = Prev
                    t = 84
                    model.add(tf.keras.layers.Dense(t ** 2, activation='relu'))
                elif (Mgen[i] == '111'):
                    # Sgen = Prev
                    t = 84
                    model.add(tf.keras.layers.Dense(t ** 2, activation='softmax'))
                    model.add(tf.keras.layers.Flatten())
                    break
                elif (Mgen[i] == '000'):
                    Kernel_Size = self.Calc_Conv_HP(Prev, Post)
                    model.add(tf.keras.layers.Conv2D(16,kernel_size=(Kernel_Size, Kernel_Size),
                                                         activation='relu'))
                elif (Mgen[i] == '001'):
                    Strides, pooling_size = self.Calc_Pool_HP(Prev, Post)
                    model.add(tf.keras.layers.AveragePooling2D(pool_size=(pooling_size,pooling_size),
                    strides=(Strides,Strides)))
                elif (Mgen[i] == '100'):
                    Kernel_Size = self.Calc_Conv_HP(Prev, Post)
                    model.add(tf.keras.layers.Conv2D(16,kernel_size=(Kernel_Size, Kernel_Size),
                                                         activation='sigmoid'))
                elif (Mgen[i] == '101'):
                    model.add(tf.keras.layers.GaussianNoise(1))
                    Kernel_Size = self.Calc_Conv_HP(Prev,Post)
                    model.add(tf.keras.layers.Conv2D(16,kernel_size=(Kernel_Size, Kernel_Size),
                                                         activation='relu'))
                else:
                    pass

            else:
                if (Mgen[i] == '010'):
                    # Sgen = Prev
                    t = 84
                    model.add(tf.keras.layers.Dense(t ** 2, activation='relu'))
                elif (Mgen[i] == '111'):
                    # Sgen = Prev
                    t = 84
                    model.add(tf.keras.layers.Dense(t ** 2, activation='softmax'))
                    model.add(tf.keras.layers.Flatten())
                    break
                elif (Mgen[i] == '000'):
                    kernel_size = self.Calc_ConvTranspose_HP(Prev,Post)
                    model.add(tf.keras.layers.Conv2DTranspose(16,kernel_size=(kernel_size,kernel_size),
                    activation='relu'))
                elif (Mgen[i] == '001'):
                    Size = self.Calc_UpSampling_HP(Prev,Post)
                    model.add(tf.keras.layers.UpSampling2D(size=(Size,Size)))
                elif (Mgen[i] == '100'):
                    kernel_size = self.Calc_ConvTranspose_HP(Prev,Post)
                    model.add(tf.keras.layers.Conv2DTranspose(16,kernel_size=(kernel_size,kernel_size),
                    activation='sigmoid'))
                elif (Mgen[i] == '101'):
                    model.add(tf.keras.layers.GaussianNoise(1))
                    kernel_size = self.Calc_ConvTranspose_HP(Prev,Post)
                    model.add(tf.keras.layers.Conv2DTranspose(16,kernel_size=(kernel_size,kernel_size),
                    activation='relu'))
                else:
                    pass
            print('added')
        for i in range(1,len(Mgen)-1):
            Prev = int(Sgen[i],16); Post = int(Sgen[i+1],16)
            j = 2
            while(((Post==0)or(Post==255))==1):
                Post = int(Sgen[i+j],16)
                j += 1
            
            print(Prev,Post)
            if (Prev >= Post):
                if (Mgen[i] == '010'):
                    # Sgen = Prev
                    t = Prev
                    model.add(tf.keras.layers.Dense(t ** 2, activation='relu'))
                elif (Mgen[i] == '111'):
                    # Sgen = Prev
                    t = Prev
                    model.add(tf.keras.layers.Dense(t ** 2, activation='softmax'))
                    model.add(tf.keras.layers.Flatten())
                    break
                elif (Mgen[i] == '000'):
                    Kernel_Size = self.Calc_Conv_HP(Prev, Post)
                    model.add(tf.keras.layers.Conv2D(16,kernel_size=(Kernel_Size, Kernel_Size),
                                                         activation='relu'))
                elif (Mgen[i] == '001'):
                    Strides, pooling_size = self.Calc_Pool_HP(Prev, Post)
                    model.add(tf.keras.layers.AveragePooling2D(pool_size=(pooling_size,pooling_size),
                    strides=(Strides,Strides)))
                elif (Mgen[i] == '100'):
                    Kernel_Size = self.Calc_Conv_HP(Prev, Post)
                    model.add(tf.keras.layers.Conv2D(16,kernel_size=(Kernel_Size, Kernel_Size),
                                                         activation='sigmoid'))
                elif (Mgen[i] == '101'):
                    model.add(tf.keras.layers.GaussianNoise(1))
                    Kernel_Size = self.Calc_Conv_HP(Prev,Post)
                    model.add(tf.keras.layers.Conv2D(16,kernel_size=(Kernel_Size, Kernel_Size),
                                                         activation='relu'))
                else:
                    pass

            else:
                if (Mgen[i] == '010'):
                    # Sgen = Prev
                    t = Prev
                    model.add(tf.keras.layers.Dense(t ** 2, activation='relu'))
                elif (Mgen[i] == '111'):
                    # Sgen = Prev
                    t = Prev
                    model.add(tf.keras.layers.Dense(t ** 2, activation='softmax'))
                    model.add(tf.keras.layers.Flatten())
                    break
                elif (Mgen[i] == '000'):
                    kernel_size = self.Calc_ConvTranspose_HP(Prev,Post)
                    model.add(tf.keras.layers.Conv2DTranspose(16,kernel_size=(kernel_size,kernel_size),
                    activation='relu'))
                elif (Mgen[i] == '001'):
                    Size = self.Calc_UpSampling_HP(Prev,Post)
                    model.add(tf.keras.layers.UpSampling2D(size=(Size,Size)))
                elif (Mgen[i] == '100'):
                    kernel_size = self.Calc_ConvTranspose_HP(Prev,Post)
                    model.add(tf.keras.layers.Conv2DTranspose(16,kernel_size=(kernel_size,kernel_size),
                    activation='sigmoid'))
                elif (Mgen[i] == '101'):
                    model.add(tf.keras.layers.GaussianNoise(1))
                    kernel_size = self.Calc_ConvTranspose_HP(Prev,Post)
                    model.add(tf.keras.layers.Conv2DTranspose(16,kernel_size=(kernel_size,kernel_size),
                    activation='relu'))
                else:
                    pass
            print('added')
        Post = int(Sgen[10],16)
        model.add(tf.keras.layers.Dense(Post ** 2, activation='softmax'))
        model.add(tf.keras.layers.Flatten())
        
        model.build(input_shape=(None,84,84,4)); model.summary()
        
        model.compile(
            loss=lambda targ, pred: tf.compat.v1.losses.huber_loss(labels=targ, predictions=pred),
            metrics=['accuracy', 'mae'], optimizer=self.optim)
        Mgen = ''.join(Mgen); model.save(Mgen+'.h5')
        print('MODEL Constructing has DONE')
        return model


    def optimizer(self):
        action = tf.placeholder('int32', (None,))
        target = tf.placeholder('float32', (None,))
        pred = self.model.output
        action_onehot = tf.one_hot(action, self.output_size)
        qval = tf.sum(pred*action_onehot, axis=1)
        error = tf.abs(target-qval)
        clipped_err = tf.clip(error, 0,1)
        overflowed_err = error - clipped_err
        loss = tf.mean(0.5*tf.sqaure(clipped_err) + overflowed_err)
        optimizer = self.optim
        grad = optimizer.get_updates(self.model.trainable_weights, [], loss)
        trainer = tf.function([self.model.input, action, target], [loss], updates=grad)

        return trainer

    def train(self, pred, action, target):
        self.trainer([pred, action, target])










