import matplotlib.pyplot as plt
import pickle

# epoch = 100

# for num in range(10, 79):
#     if (num+1) % 10 == 0:
with open(f"./result/test_epoch_19.pkl","rb") as f:
    image = pickle.load(f)
    plt.imshow(image.reshape(28,28),cmap='gist_yarg')
    plt.savefig('./result/test_epoch_19.png', dpi=200, facecolor="white")
with open(f"./result/test_epoch_29.pkl","rb") as f:
    image = pickle.load(f)
    plt.imshow(image.reshape(28,28),cmap='gist_yarg')
    plt.savefig('./result/test_epoch_29.png', dpi=200, facecolor="white")
with open(f"./result/test_epoch_39.pkl","rb") as f:
    image = pickle.load(f)
    plt.imshow(image.reshape(28,28),cmap='gist_yarg')
    plt.savefig('./result/test_epoch_39.png', dpi=200, facecolor="white")