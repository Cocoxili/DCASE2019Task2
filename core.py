from data_loader import *
from util import *
from torch.optim import lr_scheduler
import time


def create_plot_window(vis, xlabel, ylabel, title):
    return vis.line(X=np.array([1]), Y=np.array([np.nan]),
                    opts=dict(xlabel=xlabel,
                              ylabel=ylabel,
                              # showlegend=True,
                              title=title))


def train_on_fold(model, train_criterion, val_criterion,
                  optimizer, train_loader, val_loader, config, fold, vis):

    loss_window = create_plot_window(vis, '#Epochs', 'Loss', 'Train and Val Loss')
    val_lwlrap_window = create_plot_window(vis, '#Epochs', 'lwlrap', 'Validation lwlrap')
    # learning_rate_window = create_plot_window(vis, '#Epochs', 'learning rate', 'Learning rate')

    win = {'loss': loss_window,
           'val_lwlrap': val_lwlrap_window,
           # 'lr': learning_rate_window
           }

    model.train()

    best_lwlrap = 0

    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.1)  # for wave
    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 75, 90, 105], gamma=0.7)  # for logmel
    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 140], gamma=0.1)  # for MTO-resnet
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.eta_min)
    # exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    for epoch in range(config.epochs):
        exp_lr_scheduler.step()

        # train for one epoch
        train_one_epoch(train_loader, model, train_criterion, optimizer, config, fold, epoch, vis, win)

        # evaluate on validation set
        lwlrap = val_on_fold(model, val_criterion, val_loader, config, epoch, vis, win)

        # remember best prec@1 and save checkpoint
        if not config.debug:
            is_best = lwlrap > best_lwlrap
            best_lwlrap = max(lwlrap, best_lwlrap)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': config.arch,
                # 'model': model,
                'state_dict': model.state_dict(),
                'best_lwlrap': best_lwlrap,
                # 'optimizer': optimizer.state_dict(),
            }, is_best, fold, config,
            filename=config.model_dir + '/checkpoint.pth.tar')

    logging.info(' *** Best lwlrap {lwlrap:.3f}'
                 .format(lwlrap=best_lwlrap))
    return best_lwlrap


def train_one_epoch(train_loader, model, criterion, optimizer, config, fold, epoch, vis, win):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        if config.mixup:
            one_hot_labels = make_one_hot(target)
            input, target = mixup(input, one_hot_labels, alpha=3)

        # measure data loading time
        data_time.update(time.time() - end)

        if config.cuda:
            input, target = input.cuda(), target.cuda(non_blocking=True)

        # Compute output
        # print("input:", input.size(), input.type())  # ([batch_size, 1, 64, 150])
        output = model(input)
        # print("output:", output.size(), output.type())  # ([bs, 41])
        # print("target:", target.size(), target.type())  # ([bs, 41])
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            logging.info('F{fold} E{epoch} lr:{lr:.2e} '
                         'Time {batch_time.val:.1f}({batch_time.avg:.1f}) '
                         'Data {data_time.val:.1f}({data_time.avg:.1f}) '
                         'Loss {loss.avg:.3f}'.format(
                            i, len(train_loader), fold=fold, epoch=epoch,
                            lr=optimizer.param_groups[0]['lr'], batch_time=batch_time,
                            data_time=data_time, loss=losses))

    vis.line(X=np.array([epoch]), Y=np.array([losses.avg]), name='train_loss',
             win=win['loss'], update='append')
    # vis.line(X=np.array([epoch]), Y=np.array([optimizer.param_groups[0]['lr']]),
    #          win=win['lr'], update='append')


def val_on_fold(model, criterion, val_loader, config, epoch, vis, win):
    losses = AverageMeter()

    pred_all = torch.zeros(1, config.num_classes)
    target_all = torch.zeros(1, config.num_classes)

    if config.cuda:
        pred_all, target_all = pred_all.cuda(), target_all.cuda()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if config.cuda:
                input, target = input.cuda(), target.cuda(non_blocking=True)

            target_all = torch.cat((target_all, target))

            # compute output
            output = model(input)
            pred_all = torch.cat((pred_all, output))
            loss = criterion(output, target)
            # record loss
            losses.update(loss.item(), input.size(0))

    # measure elapsed time
    test_time = time.time() - end

    logging.info('Test. '
                 'Time {test_time.val:.1f} '
                 'Loss {loss.avg:.3f}'.format(test_time=test_time, loss=losses))

    lwlrap = calculate_lwlrap(target_all.cpu().numpy(), pred_all.cpu().numpy())
    logging.info(' * lwlrap {lwlrap:.3f}'.format(lwlrap=lwlrap))

    vis.line(X=np.array([epoch]), Y=np.array([losses.avg]), name='val_loss',
             win=win['loss'], update='append')
    vis.line(X=np.array([epoch]), Y=np.array([lwlrap]), name='lwlrap',
             win=win['val_lwlrap'], update='append')

    return lwlrap


def mixup(data, one_hot_labels, alpha=1):
    batch_size = data.size()[0]

    weights = np.random.beta(alpha, alpha, batch_size)

    weights = torch.from_numpy(weights).type(torch.FloatTensor)

    #  print('Mixup weights', weights)
    index = np.random.permutation(batch_size)
    x1, x2 = data, data[index]

    x = torch.zeros_like(x1)
    for i in range(batch_size):
        for c in range(x.size()[1]):
            x[i][c] = x1[i][c] * weights[i] + x2[i][c] * (1 - weights[i])

    y1 = one_hot_labels
    y2 = one_hot_labels[index]

    y = torch.zeros_like(y1)

    for i in range(batch_size):
        y[i] = y1[i] * weights[i] + y2[i] * (1 - weights[i])

    return x, y
