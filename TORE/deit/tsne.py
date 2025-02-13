import cne
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# load MNIST
mnist_train = torchvision.datasets.MNIST(train=True,
                                         download=True,
                                         transform=None)
x_train, y_train = mnist_train.data.float().numpy(), mnist_train.targets

mnist_test = torchvision.datasets.MNIST(train=False,
                                        download=True,
                                        transform=None)
x_test, y_test = mnist_test.data.float().numpy(), mnist_test.targets

x = np.concatenate([x_train, x_test], axis=0)
x = x.reshape(x.shape[0], -1)
y = np.concatenate([y_train, y_test], axis=0)

# parametric NCVis
embedder_ncvis = cne.CNE(loss_mode="nce",
                         k=15,
                         optimizer="adam",
                         parametric=True,
                         print_freq_epoch=10)
embd_ncvis = embedder_ncvis.fit_transform(x)

# non-parametric Neg-t-SNE
embedder_neg = cne.CNE(loss_mode="neg",
                       k=15,
                       optimizer="sgd",
                       momentum=0.0,
                       parametric=False,
                       print_freq_epoch=10)
embd_neg = embedder_neg.fit_transform(x)

# plot embeddings
plt.figure()
plt.scatter(*embd_ncvis.T, c=y, alpha=0.5, s=1.0, cmap="tab10", edgecolor="none")
plt.gca().set_aspect("equal")
plt.axis("off")
plt.title("Parametric NCVis of MNIST")
plt.show()