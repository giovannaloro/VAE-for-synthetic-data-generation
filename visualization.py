import matplotlib.pyplot as plt



def plot_label_clusters_3d(vae, data, labels):
    # display a 3D plot of the operation classes in the latent space 
    z_mean, _, _ = vae.encode(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(z_mean[:, 0])
    y = np.array(z_mean[:, 1])
    z = np.array(z_mean[:, 2])
    ax.scatter(x,y,z, marker="s", c=labels, s=40)
    plt.show()
    

def plot_label_clusters_2d(vae, data, labels):
    # display a 2D plot of the operation classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data, verbose=0)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()