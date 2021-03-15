import pandas as pd
import numpy as np
import pickle

import spacy
from gensim.models import FastText
from gensim.models import KeyedVectors

import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import layers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, Conv1D, MaxPooling1D, GlobalMaxPooling1D, \
    MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate, BatchNormalization
from keras.layers.core import Reshape
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, Adadelta, Adagrad
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers

from keras.losses import MeanAbsoluteError

from matplotlib import pyplot

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef


class cnn:

    def __init__(self, mutated_dst):

        self.mutated = mutated_dst

        self.train_set, self.val_set, self.test_set = np.split(self.mutated.sample(frac=1),
                                                               [int(.6 * len(self.mutated)),
                                                                int(.8 * len(self.mutated))])

        self.y_train = self.train_set['consistent'].astype(int)

        self.y_test = self.test_set['consistent'].astype(int)

        self.y_val = self.val_set['consistent'].astype(int)

    def fastext(self):

        tasks_train_w2v = [row for row in self.train_set['task_complete']]
        vector_size = 100
        w2v_model_tasks = FastText(tasks_train_w2v, sg=0, size=vector_size, window=6, min_count=1, workers=4, iter=1000)

        return w2v_model_tasks, vector_size

    @staticmethod
    def list_to_string(lst):

        one_string = ' '.join(lst)

        return one_string

    def prepare(self):

        self.train_set['task_complete_one_string'] = self.train_set['task_complete'].apply(
            lambda x: self.list_to_string(x))
        self.test_set['task_complete_one_string'] = self.test_set['task_complete'].apply(
            lambda x: self.list_to_string(x))
        self.val_set['task_complete_one_string'] = self.val_set['task_complete'].apply(lambda x: self.list_to_string(x))

        tokenizer_train = Tokenizer(lower=False)
        tokenizer_train.fit_on_texts(self.train_set['task_complete'])

        tokenizer_test = Tokenizer(lower=False)
        tokenizer_test.fit_on_texts(self.test_set['task_complete'])

        tokenizer_val = Tokenizer(lower=False)
        tokenizer_val.fit_on_texts(self.val_set['task_complete'])

        tasks_train_tokens = tokenizer_train.texts_to_sequences(self.train_set['task_complete_one_string'])
        tasks_test_tokens = tokenizer_test.texts_to_sequences(self.test_set['task_complete_one_string'])
        tasks_val_tokens = tokenizer_val.texts_to_sequences(self.val_set['task_complete_one_string'])

        num_tokens = [len(tokens) for tokens in tasks_train_tokens]
        num_tokens = np.array(num_tokens)

        max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
        max_tokens = int(max_tokens)

        tasks_train_pad = pad_sequences(tasks_train_tokens, maxlen=max_tokens, padding='post')
        tasks_test_pad = pad_sequences(tasks_test_tokens, maxlen=max_tokens, padding='post')
        tasks_val_pad = pad_sequences(tasks_val_tokens, maxlen=max_tokens, padding='post')

        return tasks_train_pad, tasks_test_pad, tasks_val_pad, max_tokens, tokenizer_train

    def create_emb_matrix(self, max_tokens, vector_size, w2v_model, tokenizertrain):

        embedding_dim = max_tokens
        embedding_size = vector_size
        num_words = len(tokenizertrain.word_index) + 1
        embedding_matrix = np.random.uniform(-1, 1, (num_words, embedding_size))

        for word, i in tokenizertrain.word_index.items():
            if i < num_words:
                embedding_vector = w2v_model[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix, num_words

    def model_train(self, num_words, vector_size, embedding_matrix, max_tokens):

        # Model parameters
        num_filters = 64
        weight_decay = 1e-4
        num_classes = 2

        tf.keras.backend.clear_session()
        cnn_model = Sequential()

        cnn_model.add(Embedding(input_dim=num_words,
                                output_dim=vector_size,
                                weights=[embedding_matrix],
                                input_length=max_tokens,
                                trainable=True,  # the layer is trained
                                name='embedding_layer'))
        cnn_model.add(layers.Conv1D(filters=num_filters, kernel_size=max_tokens, activation='relu', padding='same',
                                    kernel_regularizer=regularizers.l2(weight_decay)))
        cnn_model.add(layers.MaxPooling1D(2))
        cnn_model.add(Dropout(0.25))

        cnn_model.add(
            layers.Conv1D(filters=num_filters + num_filters, kernel_size=max_tokens, activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay)))
        cnn_model.add(layers.GlobalMaxPooling1D())
        cnn_model.add(Dropout(0.25))

        cnn_model.add(layers.Flatten())
        cnn_model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        # model.add(BatchNormalization())
        cnn_model.add(layers.Dense(num_classes, activation='softmax'))

        #         adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
        cnn_model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=sgd, metrics=['accuracy'])
        cnn_model.summary()

        return cnn_model

    def run_cnn(self):

        batch_size = 256

        fastext, emb_size = self.fastext()
        print('fastext done')
        print('==============')

        train_tasks_pad, test_tasks_pad, val_tasks_pad, tokens_max, train_tokenizer = self.prepare()
        print('Prepare done')
        print('==============')

        embed_matrix, num_of_words = self.create_emb_matrix(tokens_max, emb_size, fastext, train_tokenizer)
        print('Embedding matrix done')
        print('==============')

        cnn_model = self.model_train(num_of_words, emb_size, embed_matrix, tokens_max)
        print('CNN model declaration done')
        print('==============')

        print('Train tasks pad shape: ', train_tasks_pad.shape)
        print('Y Train shape: ', self.y_train.shape)
        print('Test tasks pad shape: ', test_tasks_pad.shape)
        print('Y Test shape: ', self.y_test.shape)

        #         train_tasks_pad=train_tasks_pad.reshape(train_tasks_pad.shape[0],train_tasks_pad.shape[1],1)
        #         test_tasks_pad=test_tasks_pad.reshape(test_tasks_pad.shape[0],test_tasks_pad.shape[1],1)

        histry = cnn_model.fit(train_tasks_pad, to_categorical(self.y_train),
                               validation_data=(test_tasks_pad, to_categorical(self.y_test)), epochs=20,
                               shuffle=True, verbose=1, batch_size=batch_size)

        return histry, cnn_model, val_tasks_pad

    def viz_loss_acc(self, history, module='all'):

        pyplot.plot(history.history['accuracy'])
        pyplot.plot(history.history['val_accuracy'])
        pyplot.title('model accuracy')
        pyplot.ylabel('accuracy')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'test'], loc='upper left')

        path_acc = 'accs/cnn_fastext_acc_' + module + '.jpg'
        pyplot.savefig(path_acc)
        pyplot.clf()

        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.title('model loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'test'], loc='upper left')

        path_loss = 'loss/cnn_fastext_loss_' + module + '.jpg'
        pyplot.savefig(path_loss)
        pyplot.clf()

    def get_metrics(self, model_cnn, val_tasks_padded, module='all'):

        #         val_tasks_padded=val_tasks_padded.reshape(val_tasks_padded.shape[0],val_tasks_padded.shape[1],1)
        score = model_cnn.evaluate(val_tasks_padded, to_categorical(self.y_val), verbose=0)
        print('Model loss:', score[0])
        print('Validation accuracy:', score[1])
        acc = score[1]

        y_pred = model_cnn.predict_classes(val_tasks_padded)

        matt = matthews_corrcoef(self.y_val, y_pred)

        cm = confusion_matrix(self.y_val, y_pred)
        print(cm)
        tp = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[0][0]
        all_val = tp + fp + fn + tn

        precision = round(tp / (tp + fp), 2)
        recall = round(tp / (tp + fn), 2)
        f1_score = (2 * precision * recall) / (precision + recall)

        precision_neg = round(tn / (tn + fn), 2)
        recall_neg = round(tn / (tn + fp), 2)
        f1_score_neg = (2 * precision_neg * recall_neg) / (precision_neg + recall_neg)

        ns_probs = [0 for _ in range(len(self.y_val))]

        ns_auc = roc_auc_score(self.y_val, ns_probs)
        lr_auc = roc_auc_score(self.y_val, y_pred)

        print('Random choice: ROC AUC=%.3f' % (ns_auc))
        print('Our model: ROC AUC=%.3f' % (lr_auc))

        ns_fpr, ns_tpr, _ = roc_curve(self.y_val, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(self.y_val, y_pred)

        pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Choice')
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Our Model')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        path = 'rocs/cnn_fastext_roc_' + module + '.jpg'
        pyplot.savefig(path)
        pyplot.clf()

        return matt, acc, precision, recall, f1_score, precision_neg, recall_neg, f1_score_neg, lr_auc


def get_filtered_mutated(mt, mod):
    mt = mt[mt['mod_keys_found_string'] == mod]

    return mt


if __name__ == "__main__":

    with open('mutated.pkl', 'rb') as input_file:
        mutated = pickle.load(input_file)

    with open('top10_list.pkl', 'rb') as input_file:
        top10_list = pickle.load(input_file)

    metrics = ['acc',
               'precision',
               'recall',
               'f1_score',
               'precision_neg_cl',
               'recall_neg_cl',
               'f1_score_neg',
               'matthews',
               'roc_auc']

    top10_dict = dict((k, '') for k in top10_list)
    results_df = pd.DataFrame(top10_dict, index=metrics)

    mutated['consistent'] = mutated['consistent'].astype(int)

    for module in top10_list:
        filtered_mutated = get_filtered_mutated(mutated, module)

        obj = cnn(filtered_mutated)

        history, cnn_model, validation_tasks_pad = obj.run_cnn()

        obj.viz_loss_acc(history, module)
        print('Visualizations done')

        matthews, accuracy, precision, recall, f1_score, precision_neg, recall_neg, f1_score_neg, roc_auc = obj.get_metrics(
            cnn_model,
            validation_tasks_pad,
            module)
        print('Metrics calculated for module ', module)
        print(matthews, accuracy, precision, recall, f1_score, precision_neg, recall_neg, f1_score_neg)

        mod_results = [accuracy, precision, recall, f1_score, precision_neg, recall_neg, f1_score_neg, matthews,
                       roc_auc]

        results_df[module] = mod_results
        print('Successfully put in results dataframe')
        print('Next module calculation')
        print('=========================')

    print(results_df)
    results_df.to_excel('cnn_fastext.xlsx')