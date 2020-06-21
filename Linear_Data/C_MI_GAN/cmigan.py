import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob
from math import log
import sys
import pickle
import os

eps = 1e-6
SEED = None
np.random.seed(seed=SEED)


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return


class TrainingDataGenerator:
    def __init__(self, cat, num_th, dz):
        self.xyz_vec = np.load(f'../data/cat{cat}/data.{num_th}k.dz{dz}.seed0.npy').astype(np.float32)
        self.ksg_mi = np.load(f'../data/cat{cat}/ksg_gt.dz{dz}.npy').astype(np.float32)[0]
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
parser.add_argument('--cat', type=str, default='F', choices=['F', 'G'])
parser.add_argument('--dx', type=int, default=1)
parser.add_argument('--dy', type=int, default=1)
parser.add_argument('--dz', type=int, default=1, choices=[1, 10, 20, 50, 100, 200])
parser.add_argument('--num_th', type=int, default=20, choices=[5, 10, 20, 50])
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--plot_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=5000)
parser.add_argument('--training_steps', type=int, default=30000)
parser.add_argument('--exp_no', type=int, default=1)

args = parser.parse_args()

batch_size = args.batch_size

params = {'gen_fc_arch': [256, 64], 'disc_fc_arch': [128, 32],
          'batch_size': batch_size, 'final_layer_neuron': args.dy}

create_directory(dir_name=f'./run{args.exp_no}/')

noise_dim = 40
checkpoint1 = 1500
checkpoint2 = 5000

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


# Training loop

data_generator = TrainingDataGenerator(args.cat, args.num_th, args.dz)

training_steps = args.training_steps
est_cmi_buf = []
steps_buf = []
gen_lr_decay = 1
disc_lr_decay = 1
disc_training_ratio = 5
gen_training_ratio = 1
true_cmi = data_generator.ksg_mi
factor = 2
for step in range(training_steps):
    if step == checkpoint1:
        disc_training_ratio = 3
    if step == checkpoint2:
        disc_training_ratio = 1
    for _ in range(disc_training_ratio):
        xyz_batch = data_generator.get_batch(batch_size)
        xz_batch = np.delete(xyz_batch, np.arange(args.dx, args.dx + args.dy), 1)
        z_batch = np.delete(xyz_batch, np.arange(0, args.dx + args.dy), 1)
        noise_vec = np.random.normal(0., 1., [batch_size, noise_dim]).astype(np.float32)
        noise_z = np.concatenate((noise_vec, z_batch), axis=1)
        disc_loss = disc_train_step(noise_z, xyz_batch, xz_batch, disc_lr_decay)
    for _ in range(gen_training_ratio):
        gen_loss = gen_train_step(noise_z, xz_batch, gen_lr_decay)
    if step > 0 and step % 1000 == 0:
        gen_lr_decay = gen_lr_decay / (5 * factor)
        disc_lr_decay = disc_lr_decay / (5 * factor)
    if step % (args.plot_interval / 10) == 0:
        est_cmi_buf.append(-disc_loss.numpy())
        steps_buf.append(step)
        print(f"Cat: {args.cat}, num_th: {args.num_th}, dz: {args.dz}, "
              f"Current Iteration: {step}, Gen Loss: {gen_loss.numpy()}, "
              f"Estimated CMI: {est_cmi_buf[-1]}, True CMI: {true_cmi}")
    if step > 0 and step % args.plot_interval == 0:
        plt.close('all')
        fig, ax = plt.subplots()
        ax.plot(steps_buf, est_cmi_buf, label='MIGAN Estimate')
        ax.plot([0, steps_buf[-1]], [true_cmi, true_cmi], label='True CMI')
        ax.set_xlabel('Training Steps')
        ax.legend(loc='best')
        fig.savefig(f'./run{args.exp_no}/cat{args.cat}.{args.num_th}k.dz{args.dz}.seed0.png')

with open(f'./run{args.exp_no}/cat{args.cat}.{args.num_th}k.dz{args.dz}.seed0.steps.migan.txt', 'wb') as fp:
    pickle.dump(steps_buf, fp)
with open(f'./run{args.exp_no}/cat{args.cat}.{args.num_th}k.dz{args.dz}.seed0.est_cmi.migan.txt', 'wb') as fp:
    pickle.dump(est_cmi_buf, fp)
