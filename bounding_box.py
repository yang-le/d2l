import torch
import d2l


if __name__ == "__main__":
    d2l.set_figsize()
    img = d2l.plt.imread('../img/catdog.jpg')
    fig = d2l.plt.imshow(img)
    # bbox是边界框的英文缩写
    dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
    fig.axes.add_patch(d2l.bbox_to_rect(dog_bbox, 'blue'))
    fig.axes.add_patch(d2l.bbox_to_rect(cat_bbox, 'red'))
    d2l.plt.show()
