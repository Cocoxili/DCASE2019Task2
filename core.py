from data_loader import *
from util import *
from torch.optim import lr_scheduler
import time


def create_plot_window(vis, xlabel, ylabel, ytickmin, ytickmax, title):
    return vis.line(X=np.array([1]), Y=np.array([np.nan]),
                    opts=dict(xlabel=xlabel,
                              ylabel=ylabel,
                              # showlegend=True,
                              ytickmin=ytickmin,
                              ytickmax=ytickmax,
                              title=title))


def train_on_fold(model, train_criterion, val_criterion,
                  optimizer, curated_df, noisy_df, val_set, X, config, fold, vis):

    loss_window = create_plot_window(vis, '#Epochs', 'Loss', 0, 0.08, 'Train and Val Loss')
    val_lwlrap_window = create_plot_window(vis, '#Epochs', 'lwlrap', 0, 0.9, 'Validation lwlrap')
    # learning_rate_window = create_plot_window(vis, '#Epochs', 'learning rate', 'Learning rate')

    win = {'loss': loss_window,
           'val_lwlrap': val_lwlrap_window,
           # 'lr': learning_rate_window
           }

    model.train()

    lwlrap = 0
    best_lwlrap = 0
    lowest_val_loss = 666.0

    def lr_lamda_fn(epoch):
        scale = 1.0
        if epoch > 30:
            scale = 0.5
        if epoch > 60:
            scale = 0.25
        if epoch > 90:
            scale = 0.1
        return scale
    # after_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.1)  # for wave
    # after_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 100], gamma=0.5)  # for logmel
    # after_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 140], gamma=0.1)  # for MTO-resnet
    after_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.eta_min)
    # after_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=config.eta_min)
    # after_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.50, patience=5, verbose=True)
    # after_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lamda_fn)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=5, after_scheduler=after_scheduler)

    for epoch in range(config.epochs):
        # exp_lr_scheduler.step()
        # exp_lr_scheduler.step(lwlrap)
        scheduler.step()

        # train for one epoch
        train_one_epoch(curated_df, noisy_df, X, model, train_criterion, optimizer, config, fold, epoch, vis, win)

        # evaluate on validation set
        lwlrap, val_loss = val_on_fold(model, val_set, X, val_criterion, config, epoch, vis, win)

        # remember best prec@1 and save checkpoint
        if not config.debug:
            is_best = lwlrap > best_lwlrap
            best_lwlrap = max(lwlrap, best_lwlrap)
            if is_best:
                best_epoch = epoch
                best_name = config.model_dir + '/model_best.' + str(fold) + '.pth.tar'
                save_checkpoint({
                    'epoch': epoch,
                    'arch': config.arch,
                    # 'model': model,
                    'state_dict': model.state_dict(),
                    'best_lwlrap': best_lwlrap,
                    # 'optimizer': optimizer.state_dict(),
                }, best_name)

        if config.early_stopping:
            is_early = val_loss < lowest_val_loss
            lowest_val_loss = min(val_loss, lowest_val_loss)
            if is_early:
                early_epoch = epoch
                early_name = config.model_dir + '/early.' + str(fold) + '.pth.tar'
                save_checkpoint({
                    'epoch': epoch,
                    # 'model': model,
                    'state_dict': model.state_dict(),
                    'best_lwlrap': lwlrap,
                    'lowest_val_loss': lowest_val_loss
                    }, early_name)

    logging.info('*** Best lwlrap {lwlrap:.3f} @ E{best_epoch}, early stopping @ E{early_epoch}.'
                 .format(lwlrap=best_lwlrap, best_epoch=best_epoch, early_epoch=early_epoch))

    return best_lwlrap


def train_one_epoch(curated_df, noisy_df, X, model, train_criterion, optimizer, config, fold, epoch, vis, win):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    T_max = 15
    D_max = 5
    D_min = 2

    D = 1 / 2 * (D_max - D_min) * np.sin(2 * epoch / T_max * np.pi) + 1 / 2 * (D_max + D_min)
    config.audio_duration = D
    composed_train = transforms.Compose([
                                         # RandomCut2D(config),
                                         # RandomHorizontalFlip(0.5),
                                         RandomFrequencyMask(1, config, 1, 0.2),
                                         RandomTimeMask(1, config, 1, 0.1),
                                         # RandomErasing(),
                                         # ToTensor(),
                                        ])

    # trainSet = FreesoundDominateMixupWithCurated(config=config, alpha=1, curated_df=curated_df, noisy_df=noisy_df,
    #                                   X=X, transform=composed_train)
    trainSet = FreesoundDominateMixup(config=config, alpha=1, curated_df=curated_df, noisy_df=noisy_df,
                                      X=X, transform=composed_train)
    train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, num_workers=4,
                              worker_init_fn=worker_init_fn)

    for i, (input, target) in enumerate(train_loader):

        if config.cuda:
            input, target = input.cuda(), target.cuda(non_blocking=True)

        # if config.mixup:
            # one_hot_labels = make_one_hot(target)
            # input, target = mixup(input, target, 1)
            # input, target = mixup_data(input, target, 1.5, config.cuda)

        # measure data loading time
        data_time.update(time.time() - end)

        if config.label_smoothing:
            target = label_smooth(target, 0.99, 0.01)

        # Compute output
        # print("input:", input.size(), input.type())  # ([batch_size, 1, 64, 150])
        output = model(input)
        # print("output:", output.size(), output.type())  # ([bs, num_class])
        # print("target:", target.size(), target.type())  # ([bs, num_class])
        loss = train_criterion(output, target)
        # print(loss.size())

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % config.print_freq == 0:
        #     logging.info('F{fold} E{epoch} lr:{lr:.2e} '
        #                  'Time {batch_time.val:.1f}({batch_time.avg:.1f}) '
        #                  'Data {data_time.val:.1f}({data_time.avg:.1f}) '
        #                  'Loss {loss.avg:.3f}'.format(
        #                     i, len(train_loader), fold=fold, epoch=epoch,
        #                     lr=optimizer.param_groups[0]['lr'], batch_time=batch_time,
        #                     data_time=data_time, loss=losses))

    logging.info('F{fold} E{epoch} lr:{lr:.2e} '
                 'Time {batch_time.sum:.1f} '
                 'Data {data_time.sum:.1f} '
                 'Loss {loss.avg:.3f}'
                 .format(fold=fold, epoch=epoch,
                         lr=optimizer.param_groups[0]['lr'],
                         batch_time=batch_time,
                         data_time=data_time, loss=losses))

    vis.line(X=np.array([epoch]), Y=np.array([losses.avg]), name='train_loss',
             win=win['loss'], update='append')
    # vis.line(X=np.array([epoch]), Y=np.array([optimizer.param_groups[0]['lr']]),
    #          win=win['lr'], update='append')
    return losses.avg


def val_on_fold(model, val_set, X, val_criterion, config, epoch, vis, win):
    losses = AverageMeter()

    pred_all = torch.zeros(1, config.num_classes)
    target_all = torch.zeros(1, config.num_classes)

    if config.cuda:
        pred_all, target_all = pred_all.cuda(), target_all.cuda()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    composed_val = transforms.Compose([RandomCut2D(config),
                                       # RandomFrequencyMask(1, config, 1, 30),
                                       # RandomTimeMask(1, config, 1, 30)
                                       # RandomErasing(),
                                       # ToTensor(),
                                       ])
    valSet = FreesoundLogmelVal(config=config, frame=val_set, X=X,
                                transform=composed_val,
                                tta=3)

    val_loader = DataLoader(valSet, batch_size=config.batch_size, shuffle=False, num_workers=4)

    with torch.no_grad():
        for i, (fname, input, target) in enumerate(val_loader):
            if config.cuda:
                input, target = input.cuda(), target.cuda(non_blocking=True)

            target_all = torch.cat((target_all, target))

            # compute output
            output = model(input)
            pred_all = torch.cat((pred_all, output))

            loss = val_criterion(output, target)
            # record loss
            losses.update(loss.item(), input.size(0))

    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(target_all.cpu().numpy(), pred_all.cpu().numpy())
    lwlrap = np.sum(per_class_lwlrap * weight_per_class)

    # measure elapsed time
    elapse = time.time() - end

    logging.info('Test. Time {test_time:.1f} Loss {loss.avg:.3f} *lwlrap {lwlrap:.3f}*'
                 .format(test_time=elapse, loss=losses, lwlrap=lwlrap))

    vis.line(X=np.array([epoch]), Y=np.array([losses.avg]), name='val_loss',
             win=win['loss'], update='append')
    vis.line(X=np.array([epoch]), Y=np.array([lwlrap]), name='lwlrap',
             win=win['val_lwlrap'], update='append')

    # calculate lwlrap on each class at last epoch
    if epoch == (config.epochs-1):
        lwlrap_of_class = pd.DataFrame(columns=['fname', 'lwlrap'])
        lwlrap_of_class['fname'] = config.labels
        lwlrap_of_class['lwlrap'] = per_class_lwlrap
        lwlrap_of_class = lwlrap_of_class.sort_values(by='lwlrap')
        logging.info('{}'.format(lwlrap_of_class))

    return lwlrap, losses.avg


def mixup(x, y, alpha=1.0, use_cuda=True):
    """
    Returns mixed inputs and targets
    """

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]

    return mixed_x, mixed_y
