import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob
from math import log
import sys
import pickle
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd

eps = 1e-6
SEED = None

np.random.seed(seed=SEED)


def normalize_data(data):
    data_norm = (data - np.mean(data, axis=0)) / (np.std(data, axis=0))
    return data_norm


class TrainingDataGenerator:
    def __init__(self, data_index):
        X = np.load('../data/Index{}_X.npy'.format(data_index)).astype(np.float32)
        X = normalize_data(X)

        Y = np.load('../data/Index{}_Y.npy'.format(data_index)).astype(np.float32)
        Y = normalize_data(Y)

        Z = np.load('../data/Index{}_Z.npy'.format(data_index)).astype(np.float32)
        Z = normalize_data(Z)

        self.xyz_vec = np.hstack((np.hstack((X, Y)), Z))
        self.n_samples = self.xyz_vec.shape[0]

    def get_batch(self, bs):
        data_indices = np.random.randint(0, self.n_samples, bs)
        return self.xyz_vec[data_indices]


def build_generator(**params):
    gen_fc_arch = params['gen_fc_arch']
    gen_model = tf.keras.models.Sequential(name='gen-seq-model')
    for layer, neurons in enumerate(gen_fc_arch):
        gen_model.add(tf.keras.layers.Dense(units=neurons,
                                            activation=tf.nn.relu,
                                            name=f'gen-dense-{layer}'))
    gen_model.add(tf.keras.layers.Dense(units=params['final_layer_neuron'],
                                        activation=None, name='gen-dense-final'))

    return gen_model


def build_discriminator(**params):
    disc_fc_arch = params['disc_fc_arch']
    disc_model = tf.keras.models.Sequential(name='disc-seq-model')

    for layer, neurons in enumerate(disc_fc_arch):
        disc_model.add(tf.keras.layers.Dense(units=neurons,
                                             activation=tf.nn.relu,
                                             name=f'disc-dense-{layer}'))
    disc_model.add(tf.keras.layers.Dense(units=1,
                                         activation=None,
                                         name='disc-dense-final'))

    return disc_model


def discriminator_loss(pred_xyz, pred_xz_gen_op):
    real_loss = -tf.reduce_mean(pred_xyz)
    fake_loss = tf.math.log(tf.reduce_mean(tf.math.exp(pred_xz_gen_op)) + eps)
    total_loss = real_loss + fake_loss

    return total_loss


def generator_loss(disc_output):
    return -tf.math.log(tf.reduce_mean(tf.math.exp(disc_output)) + eps)


parser = argparse.ArgumentParser()
parser.add_argument('--dx', type=int, default=1)
parser.add_argument('--dy', type=int, default=1)
parser.add_argument('--n_index', type=int, default=12)
parser.add_argument('--num_runs', type=int, default=5)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--plot_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=5000)
parser.add_argument('--training_steps', type=int, default=10000)
parser.add_argument('--csv_name', type=str, default='est_cmi.csv')

args = parser.parse_args()

df_column = ["Data Index"]
for i in range(1, args.num_runs + 1):
    df_column.append(f'run{i}')
df = pd.DataFrame(columns=df_column)

for data_index in range(args.n_index):
    data_generator = TrainingDataGenerator(data_index)
    df_entry = {"Data Index": data_index}
    for n_run in range(1, args.num_runs + 1):
        tf.keras.backend.clear_session()
        batch_size = args.batch_size

        params = {'gen_fc_arch': [256, 64], 'disc_fc_arch': [128, 32],
                  'batch_size': batch_size, 'final_layer_neuron': args.dy}

        noise_dim = 40

        generator = build_generator(**params)
        discriminator = build_discriminator(**params)
        lr = args.lr
        gen_opt = tf.keras.optimizers.RMSprop(lr)
        disc_opt = tf.keras.optimizers.RMSprop(lr)


        @tf.function
        def gen_train_step(noise, xz, lr_decay):
            with tf.GradientTape() as gen_tape:
                gen_op = generator(noise)
                x_gen_op_z = tf.concat([tf.concat([xz[:, 0:args.dx], gen_op], axis=1), xz[:, args.dx:]], axis=1)
                disc_output_for_x_gen_op_z = discriminator(x_gen_op_z)
                curr_gen_loss = lr_decay * generator_loss(disc_output_for_x_gen_op_z)
            gradients_of_gen = gen_tape.gradient(curr_gen_loss, generator.trainable_variables)
            gen_opt.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))

            return curr_gen_loss / lr_decay


        @tf.function
        def disc_train_step(noise, xyz, xz, lr_decay):
            with tf.GradientTape() as disc_tape:
                disc_output_for_xyz = discriminator(xyz)
                gen_op = generator(noise)
                x_gen_op_z = tf.concat([tf.concat([xz[:, 0:args.dx], gen_op], axis=1), xz[:, args.dx:]], axis=1)
                disc_output_for_x_gen_op_z = discriminator(x_gen_op_z)
                curr_disc_loss = lr_decay * discriminator_loss(disc_output_for_xyz, disc_output_for_x_gen_op_z)
            gradients_of_disc = disc_tape.gradient(curr_disc_loss, discriminator.trainable_variables)
            disc_opt.apply_gradients(zip(gradients_of_disc, discriminator.trainable_variables))

            return curr_disc_loss / lr_decay


        est_cmi_buf = []
        steps_buf = []
        gen_lr_decay = 1
        disc_lr_decay = 1
        disc_training_ratio = 3
        gen_training_ratio = 1
        factor = 2

        for step in range(args.training_steps):
            for _ in range(disc_training_ratio):
                xyz_batch = data_generator.get_batch(batch_size)
                xz_batch = np.delete(xyz_batch, np.arange(args.dx, args.dx + args.dy), 1)
                z_batch = np.delete(xyz_batch, np.arange(0, args.dx + args.dy), 1)
                noise_vec = np.random.normal(0., 1., [batch_size, noise_dim]).astype(np.float32)
                noise_z = np.concatenate((noise_vec, z_batch), axis=1)
                disc_loss = disc_train_step(noise_z, xyz_batch, xz_batch, disc_lr_decay)
            for _ in range(gen_training_ratio):
                gen_loss = gen_train_step(noise_z, xz_batch, gen_lr_decay)
            if step > 0 and step % 1500 == 0:
                gen_lr_decay = gen_lr_decay / (5 * factor)
                disc_lr_decay = disc_lr_decay / (5 * factor)
            if step % (args.plot_interval / 10) == 0:
                est_cmi_buf.append(-disc_loss.numpy())
                steps_buf.append(step)
                print(
                    f"Data Index: {data_index + 1}/{args.n_index}, Run: {n_run}/{args.num_runs}, step: {step}/{args.training_steps}, "
                    f"Gen Loss: {gen_loss.numpy()}, Estimated CMI: {est_cmi_buf[-1]}")

        df_entry.update({f'run{n_run}': np.mean(est_cmi_buf[200:])})

    df = df.append(df_entry, ignore_index=True)

df['Avg. Est.'] = df[df.columns[1:]].mean(axis=1)
df['Var'] = df[df.columns[1:-1]].var(axis=1)

with open(args.csv_name, 'a') as f:
    df.to_csv(f, header=f.tell() == 0)
