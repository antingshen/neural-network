import re
import matplotlib.pyplot as plt


def plot(logfile='train.log'):
    x, loss, train, val_x, val = [], [], [], [], []
    regex = re.compile(r'^ *(\d+)k? - loss: ([\d\.]+) - train: ([\d\.]+)( - val: ([\d\.]+))? *\n?')
    with open(logfile) as f:
        for i, line in enumerate(f):
            match = regex.match(line)
            x.append(int(match.group(1)))
            loss.append(float(match.group(2)))
            train.append(1-float(match.group(3)))
            val_str = match.group(5)
            if val_str is not None:
                val_x.append(int(match.group(1)))
                val.append(1-float(val_str))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    loss_line = ax1.plot(x, loss, 'r')
    ax1.set_ylim([0, 1])
    ax1.set_ylabel('err & loss')
    ax2 = ax1.twinx()
    train_line = ax2.plot(x, train)
    val_line = ax2.plot(val_x, val)
    ax2.set_ylim([0, 1])
    ax1.legend(loss_line + train_line + val_line, ['loss', 'train', 'val'], loc='upper right')
    ax1.set_xlabel('thousands of iterations')
    plt.show()
    # plt.savefig('mse.png')

if __name__ == '__main__':
    plot()
