import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def one_hot(labels):
    final = np.zeros([len(labels), max(labels)+1])
    final[np.arange(len(labels)), labels] = 1
    return final

def plot_acc_loss(dir,history):
    plt.figure()
    plt.title('Training performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss+error')
    plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.plot(history.epoch, history.history['acc'],label='train_acc')
    plt.plot(history.epoch, history.history['val_acc'], label = 'val_acc')
    plt.legend()
    plt.savefig('%s/Training_performance'%dir)
    # plt.show()
def plot_confusion_matrix(cm, labels):
    cmap = plt.cm.binary
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
    for x_test, y_test in zip(x.flatten(), y.flatten()):
        #

        if (intFlag):
            c = cm[y_test][x_test]
            plt.text(x_test, y_test, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')

        else:
            c = cm_normalized[y_test][x_test]
            if (c > 0.01):
                #这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_test, y_test, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
            else:
                plt.text(x_test, y_test, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('')
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('Index of True Classes')
    plt.xlabel('Index of Predict Classes')
    plt.savefig('confusion_matrix.jpg', dpi=300)