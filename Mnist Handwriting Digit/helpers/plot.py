import matplotlib.pyplot as plt

def display_imgs(imgs):
    img_per_row = 5
    n_rows = 2

    fig, axs = plt.subplots(n_rows, img_per_row, figsize=(img_per_row*2, n_rows*2))
    for y in range(n_rows):
        for x in range(img_per_row):
            axs[y, x].imshow(imgs[y*img_per_row + x], cmap = "gray")
            axs[y, x].axis('off')
    
    plt.show()

