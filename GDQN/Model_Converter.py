import tensorflow as tf


class GeneticModel:
    def __init__(self, Mgen, Sgen, optim=tf.keras.optimizers.Adam, lr=1e-7):
        self.lr = lr;
        self.optim = optim(learning_rate=lr)
        self.model = self.Convert2Model(Mgen, Sgen)
        pass

    def Calc_Conv_HP(self, PS_gen1, PS_gen2):
        Kernel_Size = PS_gen1 - PS_gen2 + 1
        return Kernel_Size

    def Calc_Pool_HP(self, PS_gen1, PS_gen2):
        strides = 1;
        pooling_size = 1
        if ((PS_gen1 + 1) % (PS_gen2 + 1) == 0):
            strides = int((PS_gen1 + 1) / (PS_gen2 + 1));
            pooling_size = strides
        else:
            strides = int((PS_gen1 + 2) / (PS_gen2 + 1));
            pooling_size = strides - 1

        return strides, pooling_size

    def Calc_ConvTranspose_HP(self, PS_gen1, PS_gen2):
        kernel_size = PS_gen2 - PS_gen1 + 1
        return kernel_size

    def Calc_UpSampling_HP(self, PS_gen1, PS_gen2):
        size = int(PS_gen2 / PS_gen1)
        return size

    def lambda_out_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = 1
        return tuple(shape)

    def Convert2Model(self, Mgen, Sgen):
        model = tf.keras.models.Sequential()
        Master_Gen = (lambda x: [x[3 * i:3 * i + 3] for i in range(11)])(Mgen)
        Sgen = (lambda x: [x[2 * i:2 * i + 2] for i in range(11)])(Sgen)
        Valid_Sgen_List = [];
        Valid_Mgen_List = []

        for i in range(len(Master_Gen)):
            if ((Master_Gen[i] != '011') and (Master_Gen[i] != '110')):
                Valid_Mgen_List.append(Master_Gen[i])
                Valid_Sgen_List.append(Sgen[i])
            if (Master_Gen[i] == '111'):
                break

        for i in range(len(Valid_Mgen_List)):
            Prev = 1
            try:
                Prev = model.layers[i].output_shape[1]
                Post = int(Valid_Sgen_List[i], 16)
            except IndexError:
                Post = 4
            if(i==0):
                if (Prev >= Post):
                    if (Valid_Mgen_List[i] == '010'):
                        # Sgen = Prev
                        t = Prev
                        model.add(tf.keras.layers.Dense(t ** 2, activation='relu',input_shape=(84,84,4)))

                    elif (Valid_Mgen_List[i] == '111'):
                        # Sgen = Prev
                        t = Prev
                        model.add(tf.keras.layers.Dense(t ** 2, activation='softmax',input_shape=(84,84,4)))
                        break
                    elif (Valid_Mgen_List[i] == '000'):
                        Kernel_Size = self.Calc_Conv_HP(Prev, Post)
                        model.add(tf.keras.layers.Conv2D(16, kernel_size=(Kernel_Size, Kernel_Size),
                                                         activation='relu',input_shape=(84,84,4)))

                    elif (Valid_Mgen_List[i] == '001'):
                        Strides, pooling_size = self.Calc_Pool_HP(Prev, Post)
                        model.add(tf.keras.layers.AveragePooling2D(pool_size=(pooling_size, pooling_size),
                                                                   strides=(Strides, Strides),input_shape=(84,84,4)))

                    elif (Valid_Mgen_List[i] == '100'):
                        Kernel_Size = self.Calc_Conv_HP(Prev, Post)
                        model.add(tf.keras.layers.Conv2D(16, kernel_size=(Kernel_Size, Kernel_Size),
                                                         activation='sigmoid',input_shape=(84,84,4)))

                    elif (Valid_Mgen_List[i] == '101'):
                        Kernel_Size = self.Calc_Conv_HP(Prev, Post)
                        model.add(tf.keras.layers.Conv2D(16, kernel_size=(Kernel_Size, Kernel_Size),
                                                         activation='relu',input_shape=((84,84,4))))
                        model.add(tf.keras.layers.GaussianNoise(1))


                else:
                    if (Valid_Mgen_List[i] == '010'):
                        t = Prev
                        model.add(tf.keras.layers.Dense(t ** 2, activation='relu',input_shape=((84,84,4))))

                    elif (Valid_Mgen_List[i] == '111'):
                        t = Prev
                        model.add(tf.keras.layers.Dense(t ** 2, activation='softmax',input_shape=((84,84,4))))
                        break
                    elif (Valid_Mgen_List[i] == '000'):
                        kernel_size = self.Calc_ConvTranspose_HP(Prev, Post)
                        model.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=(kernel_size, kernel_size),
                                                                  activation='relu',input_shape=((84,84,4))))

                    elif (Valid_Mgen_List[i] == '001'):
                        Size = self.Calc_UpSampling_HP(Prev, Post)
                        model.add(tf.keras.layers.UpSampling2D(size=(Size, Size),input_shape=((84,84,4))))

                    elif (Valid_Mgen_List[i] == '100'):
                        kernel_size = self.Calc_ConvTranspose_HP(Prev, Post)
                        model.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=(kernel_size, kernel_size),
                                                                  activation='sigmoid',input_shape=((84,84,4))))

                    elif (Valid_Mgen_List[i] == '101'):
                        kernel_size = self.Calc_ConvTranspose_HP(Prev, Post)
                        model.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=(kernel_size, kernel_size),
                                                                  activation='relu',input_shape=((84,84,4))))
                        model.add(tf.keras.layers.GaussianNoise(1))
            else:
                if (Prev >= Post):
                    if (Valid_Mgen_List[i] == '010'):
                        # Sgen = Prev
                        t = Prev
                        model.add(tf.keras.layers.Dense(t ** 2, activation='relu'))

                    elif (Valid_Mgen_List[i] == '111'):
                        # Sgen = Prev
                        t = Prev
                        model.add(tf.keras.layers.Dense(t ** 2, activation='softmax'))
                        break
                    elif (Valid_Mgen_List[i] == '000'):
                        Kernel_Size = self.Calc_Conv_HP(Prev, Post)
                        model.add(tf.keras.layers.Conv2D(16, kernel_size=(Kernel_Size, Kernel_Size),
                                                         activation='relu'))

                    elif (Valid_Mgen_List[i] == '001'):
                        Strides, pooling_size = self.Calc_Pool_HP(Prev, Post)
                        model.add(tf.keras.layers.AveragePooling2D(pool_size=(pooling_size, pooling_size),
                                                                   strides=(Strides, Strides)))

                    elif (Valid_Mgen_List[i] == '100'):
                        Kernel_Size = self.Calc_Conv_HP(Prev, Post)
                        model.add(tf.keras.layers.Conv2D(16, kernel_size=(Kernel_Size, Kernel_Size),
                                                         activation='sigmoid'))

                    elif (Valid_Mgen_List[i] == '101'):
                        Kernel_Size = self.Calc_Conv_HP(Prev, Post)
                        model.add(tf.keras.layers.Conv2D(16, kernel_size=(Kernel_Size, Kernel_Size),
                                                         activation='relu'))
                        model.add(tf.keras.layers.GaussianNoise(1))


                else:
                    if (Valid_Mgen_List[i] == '010'):
                        t = Prev
                        model.add(tf.keras.layers.Dense(t ** 2, activation='relu'))

                    elif (Valid_Mgen_List[i] == '111'):
                        t = Prev
                        model.add(tf.keras.layers.Dense(t ** 2, activation='softmax'))
                        break
                    elif (Valid_Mgen_List[i] == '000'):
                        kernel_size = self.Calc_ConvTranspose_HP(Prev, Post)
                        model.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=(kernel_size, kernel_size),
                                                                  activation='relu'))

                    elif (Valid_Mgen_List[i] == '001'):
                        Size = self.Calc_UpSampling_HP(Prev, Post)
                        model.add(tf.keras.layers.UpSampling2D(size=(Size, Size)))

                    elif (Valid_Mgen_List[i] == '100'):
                        kernel_size = self.Calc_ConvTranspose_HP(Prev, Post)
                        model.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=(kernel_size, kernel_size),
                                                                  activation='sigmoid'))

                    elif (Valid_Mgen_List[i] == '101'):
                        kernel_size = self.Calc_ConvTranspose_HP(Prev, Post)
                        model.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=(kernel_size, kernel_size),
                                                                  activation='relu'))
                        model.add(tf.keras.layers.GaussianNoise(1))


        model.add(tf.keras.layers.Flatten());
        model.add(tf.keras.layers.Dense(4))
        print('MODEL Constructing has DONE')
        print(type(model))
        return model