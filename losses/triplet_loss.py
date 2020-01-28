import torch
import numpy
import sys

class TripletLoss(torch.nn.modules.loss._Loss):
    """
    Triplet loss for representations of time series. Optimized for training
    sets where all time series have the same length.

    Takes as input a tensor as the chosen batch to compute the loss,
    a PyTorch module as the encoder, a 3D tensor (`B`, `C`, `L`) containing
    the training set, where `B` is the batch size, `C` is the number of
    channels and `L` is the length of the time series, as well as a boolean
    which, if True, enables to save GPU memory by propagating gradients after
    each loss term, instead of doing it after computing the whole loss.

    The triplets are chosen in the following manner. First the size of the
    positive and negative samples are randomly chosen in the range of lengths
    of time series in the dataset. The size of the anchor time series is
    randomly chosen with the same length upper bound but the the length of the
    positive samples as lower bound. An anchor of this length is then chosen
    randomly in the given time series of the train set, and positive samples
    are randomly chosen among subseries of the anchor. Finally, negative
    samples of the chosen length are randomly chosen in random time series of
    the train set.

    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty):
        super(TripletLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty

    def forward(self, batch, encoder, train, save_memory=False, sliding_window=False, lambda_0=1, lambda_1=0, lambda_2=0):
        # print(lambda_0)
        # print(lambda_1)
        # print(lambda_2)
        # sys.exit()

        '''
        added a sliding window. This sliding window will get us the x_pos given the x_ref example
        Note that, default sliding window will be off. Now if the sliding window is True, then this will be evaluated. 
        For every example in the batch, we select the x_ref. Then choose the beginning point inside the x_ref subsequence
        Choose if we can take a window of lenght same as x_ref's length from the point of x_pos. Check if left window can be taken 
        or the right window can be taken (so that we don't breach the indexes).
        '''
        batch_size = batch.size(0)
        train_size = train.size(0)
        length = min(self.compared_length, train.size(2))

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)

        samples = numpy.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )

        samples = torch.LongTensor(samples)

        # Choice of length of positive and negative samples
        length_pos_neg = numpy.random.randint(1, high=length + 1)  #this is sampling s_pos_neg from [1, length_max]

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'

        random_length = numpy.random.randint(
            length_pos_neg, high=length + 1
        )  # Length of anchors                             #sampling the lenght of reference from [s_pos_neg, length_max]
        ##print('rand length '+str(random_length))
        beginning_batches = numpy.random.randint(
            0, high=length - random_length + 1, size=batch_size
        )  # Start of anchors    choosing the reference example starting indices??? 

        # The positive samples are chosen at random in the chosen anchors
        beginning_samples_pos = numpy.random.randint(
            0, high=random_length - length_pos_neg + 1, size=batch_size
        )  # Start of positive samples in the anchors

        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + length_pos_neg


        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = numpy.random.randint(
            0, high=length - random_length + 1,
            size=(self.nb_random_samples, batch_size)
        )

        representation = encoder(torch.cat(
            [batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length
            ] for j in range(batch_size)]
        ),fmaps=True)
        x_ref_local = representation['fmap']
        x_ref_global = representation['out']

        #get strides! left stride  =  U(1, x_ref)
        # right strdide = U(1, L-x_ref_random_len)
        # choose the biggest stride among them and then make the x_pos. 
        sliding_window = True
        ##print(sliding_window)
        if sliding_window:
            for j in range(batch_size):
                s_left = -1
                s_right = -1
                left_sel = False
                h_r = 1+length-beginning_batches[j]-random_length
                h_l =  beginning_batches[j]+1
                if h_r==1:
                    s_right=0
                elif h_l==1:
                    s_left=0
                else:   
                    s_right  = numpy.random.randint(1,h_r)
                    s_left   = numpy.random.randint(1,h_l)
                if s_left >= s_right:
                    left_sel=True
                if left_sel:
                    beginning_positive[j] = beginning_batches[j]-s_left
                    end_positive[j]=beginning_batches[j] + random_length - s_left
                    ##print('left' + str(end_positive[j]-beginning_positive[j]))
                else:
                    beginning_positive[j] = beginning_batches[j] + s_right
                    end_positive[j]=beginning_batches[j] + random_length + s_right
                    ##print('right' + str(end_positive[j]-beginning_positive[j]))


        positive_representation = encoder(torch.cat(
        [batch[j: j + 1, :, beginning_positive[j]: end_positive[j]] for j in range(batch_size)]
        ),fmaps=True)  # Positive samples representations 
        x_pos_local = positive_representation['fmap']
        x_pos_global = positive_representation['out']
        size_representation = x_ref_global.size(1)
        
        loss = 0.        
        sx = x_pos_local.size(1)
        sy = x_pos_local.size(2)

        loss_GG = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            x_ref_global.view(batch_size, 1, size_representation),
            x_pos_global.view(batch_size, size_representation, 1)
        )))

        loss_LL = 0.
        for y in range(sy):
            loss_LL += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(x_pos_local[:,:,y].squeeze().view(batch_size,1,sx),
                x_ref_local[:,:,y].squeeze().view(batch_size,sx,1))))
        loss_LL = (loss_LL/sy)

        loss_GL = 0.
        for y in range(sy):
            loss_GL+= -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(x_pos_local[:,:,y].squeeze().view(batch_size,1,sx),
                x_ref_global.view(batch_size,size_representation,1)).squeeze())).double()
        
        loss_GL = (loss_GL/sy)

        loss = lambda_0*loss_GG + lambda_1*loss_GL + lambda_2*loss_LL

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        loss_GL_temp = 0.
        loss_LL_temp = 0.
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = encoder(
                torch.cat([train[samples[i, j]: samples[i, j] + 1][
                    :, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + random_length
                ] for j in range(batch_size)])
            ,fmaps=True)
            x_neg_local = negative_representation['fmap']
            x_neg_global = negative_representation['out']

            sx = x_neg_local.size(1)
            sy = x_neg_local.size(2)

            
            for y in range(sy):
                if(y==0):
                    temp_tensor = torch.nn.functional.logsigmoid(-torch.bmm(x_neg_local[:,:,y].squeeze().view(batch_size,1,sx),
                        x_ref_global.view(batch_size,size_representation,1)).squeeze()).view(1,batch_size)

                    temp2_tensor = torch.nn.functional.logsigmoid(-torch.bmm(x_neg_local[:,:,y].squeeze().view(batch_size,1,sx),
                        x_ref_local[:,:,y].view(batch_size,size_representation,1)).squeeze()).view(1,batch_size)
                else:
                    temp_tensor = torch.cat((temp_tensor,(torch.nn.functional.logsigmoid(-torch.bmm(x_neg_local[:,:,y].squeeze().view(batch_size,1,sx),
                        x_ref_global.view(batch_size,size_representation,1)).squeeze()).view(1,batch_size))),0)

                    temp2_tensor = torch.cat((temp2_tensor,(torch.nn.functional.logsigmoid(-torch.bmm(x_neg_local[:,:,y].squeeze().view(batch_size,1,sx),
                        x_ref_local[:,:,y].view(batch_size,size_representation,1)).squeeze()).view(1,batch_size))),0)
    
            if(i==0):
                loss_GL_temp = temp_tensor.view(1,sy,batch_size)
                loss_LL_temp = temp2_tensor.view(1,sy,batch_size)
            else:
                loss_GL_temp = torch.cat((loss_GL_temp,temp_tensor.view(1,sy,batch_size)),0)
                loss_LL_temp = torch.cat((loss_LL_temp,temp2_tensor.view(1,sy,batch_size)),0)

            loss_GG += -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    x_ref_global.view(batch_size, 1, size_representation),
                    x_neg_global.view(
                        batch_size, size_representation, 1
                    )
                ))
            )

        loss_GL += -(torch.mean(torch.sum(torch.sum(loss_GL_temp,dim=0),dim=0))/sy)
        loss_LL += -(torch.mean(torch.sum(torch.sum(loss_LL_temp,dim=0),dim=0))/sy)
        loss = lambda_0*loss_GG + lambda_1*loss_GL + lambda_2*loss_LL

        if save_memory and i != self.nb_random_samples - 1:
            loss.backward(retain_graph=True)
            loss = 0
            del negative_representation
            torch.cuda.empty_cache()

        return loss


class TripletLossVaryingLength(torch.nn.modules.loss._Loss):
    """
    Triplet loss for representations of time series where the training set
    features time series with unequal lengths.

    Takes as input a tensor as the chosen batch to compute the loss,
    a PyTorch module as the encoder, a 3D tensor (`B`, `C`, `L`) containing the
    training set, where `B` is the batch size, `C` is the number of channels
    and `L` is the maximum length of the time series (NaN values representing
    the end of a shorter time series), as well as a boolean which, if True,
    enables to save GPU memory by propagating gradients after each loss term,
    instead of doing it after computing the whole loss.

    The triplets are chosen in the following manner. First the sizes of
    positive and negative samples are randomly chosen in the range of lengths
    of time series in the dataset. The size of the anchor time series is
    randomly chosen with the same length upper bound but the the length of the
    positive samples as lower bound. An anchor of this length is then chosen
    randomly in the given time series of the train set, and positive samples
    are randomly chosen among subseries of the anchor. Finally, negative
    samples of the chosen length are randomly chosen in random time series of
    the train set.

    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty):
        super(TripletLossVaryingLength, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty

    def forward(self, batch, encoder, train, save_memory=False):
        batch_size = batch.size(0)
        train_size = train.size(0)
        max_length = train.size(2)

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)

        samples = numpy.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )
        samples = torch.LongTensor(samples)

        # Computation of the lengths of the relevant time series
        with torch.no_grad():
            lengths_batch = max_length - torch.sum(
                torch.isnan(batch[:, 0]), 1
            ).data.cpu().numpy()
            lengths_samples = numpy.empty(
                (self.nb_random_samples, batch_size), dtype=int
            )
            for i in range(self.nb_random_samples):
                lengths_samples[i] = max_length - torch.sum(
                    torch.isnan(train[samples[i], 0]), 1
                ).data.cpu().numpy()

        # Choice of lengths of positive and negative samples
        lengths_pos = numpy.empty(batch_size, dtype=int)
        lengths_neg = numpy.empty(
            (self.nb_random_samples, batch_size), dtype=int
        )
        for j in range(batch_size):
            lengths_pos[j] = numpy.random.randint(
                1, high=min(self.compared_length, lengths_batch[j]) + 1
            )
            for i in range(self.nb_random_samples):
                lengths_neg[i, j] = numpy.random.randint(
                    1,
                    high=min(self.compared_length, lengths_samples[i, j]) + 1
                )

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = numpy.array([numpy.random.randint(
            lengths_pos[j],
            high=min(self.compared_length, lengths_batch[j]) + 1
        ) for j in range(batch_size)])  # Length of anchors
        beginning_batches = numpy.array([numpy.random.randint(
            0, high=lengths_batch[j] - random_length[j] + 1
        ) for j in range(batch_size)])  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        # Start of positive samples in the anchors
        beginning_samples_pos = numpy.array([numpy.random.randint(
            0, high=random_length[j] - lengths_pos[j] + 1
        ) for j in range(batch_size)])
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + lengths_pos

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = numpy.array([[numpy.random.randint(
            0, high=lengths_samples[i, j] - lengths_neg[i, j] + 1
        ) for j in range(batch_size)] for i in range(self.nb_random_samples)])

        representation = torch.cat([encoder(
            batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length[j]
            ]
        ) for j in range(batch_size)])  # Anchors representations

        positive_representation = torch.cat([encoder(
            batch[
                j: j + 1, :,
                end_positive[j] - lengths_pos[j]: end_positive[j]
            ]
        ) for j in range(batch_size)])  # Positive samples representations

        size_representation = representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            representation.view(batch_size, 1, size_representation),
            positive_representation.view(batch_size, size_representation, 1)
        )))

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = torch.cat([encoder(
                train[samples[i, j]: samples[i, j] + 1][
                    :, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + lengths_neg[i, j]
                ]
            ) for j in range(batch_size)])
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    negative_representation.view(
                        batch_size, size_representation, 1
                    )
                ))
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return loss
