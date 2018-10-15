import matplotlib.pyplot as plt
def gen(img, label, model):
    for iterator in range(200):
        fake_imgs = image_generator(img, 64)
        pred = model.predict(fake_imgs, batch_size = 64)
        pred_label = np.argmax(pred, axis=1)
        flag = False
        for i in range(64):
            if pred_label[i] == label:
                choosed_img = fake_imgs[i]
                flag = True
                break
        if flag == False:
        	break
        else:
            img = choosed_img
    plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray)
    plt.show()
