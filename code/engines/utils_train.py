import os
import matplotlib.pyplot as plt


def update_dict(Losses, losses, Performances, performances):
    for key, value in losses.items():
        Losses[key].append(value)
    for key, value in performances.items():
        Performances[key].append(value)
    return Losses, Performances


def update_LRs(LRs, lr):
    LRs.append(lr)
    return LRs


def save_losses_and_performances(cfg, losses, performances, LRs):
    # Draw Losses
    fig = plt.figure()
    for key, value in losses.items():
        plt.plot(value, label=f'Loss_{key}')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(cfg.viz_dir, 'Losses.jpg'))
    plt.close()

    # Draw Performances
    fig = plt.figure()
    plt.plot(performances, label='PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('Performances')
    plt.legend(loc='upper left')

    plt.savefig(os.path.join(cfg.viz_dir, 'Performances.jpg'))
    plt.close()

    # Draw Learning rates
    fig = plt.figure()
    plt.plot(LRs, label='LR')
    plt.xlabel('Epochs')
    plt.ylabel('Learning rates')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(cfg.viz_dir, 'Learning_rate.jpg'))
    plt.close()


def save_learning_rates(cfg, LRs):
    fig = plt.figure()
    plt.plot(LRs, label='LR')
    plt.xlabel('Epochs')
    plt.ylabel('Learning rates')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(cfg.viz_dir, 'Learning_rate.jpg'))
    plt.close()



def update_and_logging(loss_t, loss_dict):
    txt = ''
    for key, value in loss_dict.items():
        loss_t[key] += value.item()
        txt += f'{key}: {value.item():.5f}  '
    return loss_t, txt


def logging(loss_t, denom):
    txt = ''
    for key, value in loss_t.items():
        loss_t[key] = value / denom
        txt += f'[{key}: {loss_t[key]:.5f}] '
    return txt
