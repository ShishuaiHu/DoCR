from docr.utils.fourier import FDA_img_to_hfi, FDA_source_to_target_np
from docr.datasets.utils.normalize import normalize_image
from docr.loss_functions.entropy_loss import ent_loss
from time import time
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data
import torch.nn as nn
from docr.utils.file_utils import check_folders
from batchgenerators.utilities.file_and_folder_operations import *
from docr.models.DoCR import DoCR
from docr.datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from docr.datasets.utils.convert_csv_to_list import convert_labeled_list
from docr.datasets.utils.transform import collate_fn_tr, collate_fn_ts
from docr.utils.lr import adjust_learning_rate
from docr.utils.metrics.dice import get_hard_dice
from sklearn.metrics import accuracy_score


def encoding_domain(root_folder, name_list):
    all_domain_str = ['Bi', 'Ma', 'ME']
    domain_str_list = [split_path(i.replace(root_folder, ''))[2][:2] for i in name_list]
    domain_encoding = np.zeros((len(domain_str_list), len(all_domain_str)))
    for i in range(len(domain_str_list)):
        domain_encoding[i][all_domain_str.index(domain_str_list[i])] = 1.0
    return torch.from_numpy(domain_encoding).cuda().to(dtype=torch.float32)


def train(args):
    model_name = args.model
    gpu = tuple(args.gpu)
    log_folder = args.log_folder
    tag = args.tag
    log_folder = join(log_folder, model_name+'_'+tag)
    patch_size = tuple(args.patch_size)
    batch_size = args.batch_size
    initial_lr = args.initial_lr
    save_interval = args.save_interval
    num_epochs = args.num_epochs
    continue_training = args.continue_training
    num_threads = args.num_threads
    root_folder = args.root
    tr_csv = tuple(args.tr_csv)
    ts_csv = tuple(args.ts_csv)
    tu_csv = tuple(args.tu_csv)
    shuffle = not args.no_shuffle
    beta = args.beta
    rec_w = args.rec_w

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu])
    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    writer = SummaryWriter(log_dir=tensorboard_folder)

    tr_img_list, tr_label_list = convert_labeled_list(tr_csv, r=1)
    tr_dataset = RIGA_labeled_set(root_folder, tr_img_list, tr_label_list, patch_size, img_normalize=False)
    tu_img_list, _ = convert_labeled_list(tu_csv, r=1)
    tu_dataset = RIGA_unlabeled_set(root_folder, tu_img_list, patch_size, img_normalize=False)
    ts_img_list, ts_label_list = convert_labeled_list(ts_csv, r=1)
    ts_dataset = RIGA_labeled_set(root_folder, ts_img_list, ts_label_list, patch_size)

    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                collate_fn=collate_fn_tr)
    tu_dataloader = torch.utils.data.DataLoader(tu_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                collate_fn=collate_fn_ts)
    tu_dataloader_iter = iter(tu_dataloader)
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads//2,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=collate_fn_ts)

    model = DoCR()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.99, nesterov=True)

    start_epoch = 0
    if continue_training:
        assert isfile(join(model_folder, 'model_latest.model')) or isfile(join(model_folder, 'model_final.model')), \
            'missing model checkpoint!'
        params = torch.load(join(model_folder, 'model_latest.model')) if isfile(join(model_folder, 'model_latest.model')) \
            else torch.load(join(model_folder, 'model_final.model'))
        model.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        start_epoch = params['epoch']
    print('start epoch: {}'.format(start_epoch))

    amp_grad_scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()
    cls_criterion = nn.CrossEntropyLoss(reduction='mean')

    start = time()
    tu_iter_counter = 0
    epoch = 0
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}:'.format(epoch))
        start_epoch = time()
        model.train()
        lr = adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs)
        print('  lr: {}'.format(lr))

        warmup = True if epoch < 20 else False

        train_loss_list = list()
        train_disc_dice_list = list()
        train_cup_dice_list = list()
        cls_loss_list = list()
        acc_list = list()
        for _, batch in enumerate(tr_dataloader):
            tu_iter_counter += 1
            if tu_iter_counter == len(tu_dataloader_iter):
                tu_dataloader_iter = iter(tu_dataloader)
                tu_iter_counter = 1
            tu_batch = tu_dataloader_iter.next()

            if warmup or np.random.random() < 0.25:
                fda_beta = 0
            else:
                fda_beta = round(np.random.random()*0.15, 2)
            lowf_batch = tu_batch['data'][0:batch['data'].shape[0]] if np.random.random() < 0.66 else np.random.permutation(batch['data'])
            src_in_trg = FDA_source_to_target_np(batch['data'], lowf_batch,
                                                 L=fda_beta)
            src_hf = FDA_img_to_hfi(batch['data'], L=beta)
            trg_hf = FDA_img_to_hfi(tu_batch['data'], L=beta)

            src_domain_encoding = encoding_domain(root_folder, batch['name'])
            trg_domain_encoding = encoding_domain(root_folder, tu_batch['name'])
            src_domain_index = src_domain_encoding.argmax(dim=1)
            trg_domain_index = trg_domain_encoding.argmax(dim=1)

            trg_hf = torch.from_numpy(normalize_image(trg_hf)).cuda().to(dtype=torch.float32)
            src_hf = torch.from_numpy(normalize_image(src_hf)).cuda().to(dtype=torch.float32)
            src_data = torch.from_numpy(normalize_image(src_in_trg)).cuda().to(dtype=torch.float32)
            trg_data = torch.from_numpy(normalize_image(tu_batch['data'])).cuda().to(dtype=torch.float32)
            seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)

            optimizer.zero_grad()
            no_fda = fda_beta == 0
            with autocast():
                trg_seg_output, trg_rec_output = model(trg_data, domain_label=trg_domain_encoding, training=True)
                trg_rec_loss = F.l1_loss(torch.tanh(trg_rec_output), trg_hf, reduction='mean')
                ent_loss_trg = ent_loss(trg_seg_output)
                if no_fda:
                    cls_data = torch.cat([src_data, trg_data], dim=0)
                    cls_domain_index = torch.cat([src_domain_index, trg_domain_index], dim=0)
                    rand_index = torch.randperm(cls_domain_index.size(0))
                    cls_data = cls_data[rand_index]
                    cls_domain_index = cls_domain_index[rand_index]
                    cls_out = model.forward_cls(cls_data)
                    cls_loss = cls_criterion(cls_out, cls_domain_index)
                    this_acc_score = accuracy_score(torch.max(torch.softmax(cls_out, dim=1), dim=1)[1].cpu().numpy(),
                                                    cls_domain_index.cpu().numpy())
                    seg_output, rec_output = model(src_data, domain_label=src_domain_encoding, training=True)
                    seg_loss = criterion(seg_output[:, 0], (seg[:, 0] > 0) * 1.0) + \
                           criterion(seg_output[:, 1], (seg[:, 0] == 2) * 1.0)
                    rec_loss = F.l1_loss(torch.tanh(rec_output), src_hf, reduction='mean')
                    loss = seg_loss + rec_w * (rec_loss + trg_rec_loss) + 0.005 * ent_loss_trg + cls_loss
                else:
                    seg_output, rec_output = model(src_data, domain_label=model.gen_domain_label(src_data), training=True)
                    seg_loss = criterion(seg_output[:, 0], (seg[:, 0] > 0) * 1.0) + \
                               criterion(seg_output[:, 1], (seg[:, 0] == 2) * 1.0)
                    rec_loss = F.l1_loss(torch.tanh(rec_output), src_hf, reduction='mean')
                    loss = seg_loss + rec_w * (rec_loss + trg_rec_loss) + 0.005 * ent_loss_trg

            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
            train_loss_list.append(seg_loss.detach().cpu().numpy())
            if no_fda:
                cls_loss_list.append(cls_loss.detach().cpu().numpy())
                acc_list.append(this_acc_score)
            output_sigmoid = torch.sigmoid(seg_output)
            train_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
            train_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
            del seg
        mean_tr_loss = np.mean(train_loss_list)
        mean_cls_loss = np.mean(cls_loss_list)
        mean_tr_acc = np.mean(acc_list)
        mean_tr_disc_dice = np.mean(train_disc_dice_list)
        mean_tr_cup_dice = np.mean(train_cup_dice_list)
        writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
        writer.add_scalar("Train Scalars/CLS Loss", mean_cls_loss, epoch)
        writer.add_scalar("Train Scalars/ACC", mean_tr_acc, epoch)
        writer.add_scalar("Train Scalars/Disc Dice", mean_tr_disc_dice, epoch)
        writer.add_scalar("Train Scalars/Cup Dice", mean_tr_cup_dice, epoch)
        print('  Tr loss: {}; CLS loss: {}\n'
              '  Tr disc dice: {}; Cup dice: {}, ACC: {}'.format(mean_tr_loss, mean_cls_loss, mean_tr_disc_dice, mean_tr_cup_dice, mean_tr_acc))

        val_loss_list = list()
        val_disc_dice_list = list()
        val_cup_dice_list = list()
        val_cls_loss_list = list()
        val_acc_list = list()
        with torch.no_grad():
            model.eval()
            for _, batch in enumerate(ts_dataloader):
                data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)

                val_domain_encoding = encoding_domain(root_folder, batch['name'])
                val_domain_index = val_domain_encoding.argmax(dim=1)
                with autocast():
                    domain_out = model.forward_cls(data)
                    output, rec_output = model(data, domain_label=val_domain_encoding, training=True)  # set training to 'True' means output reconstruction result, not really training
                    loss = criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + criterion(output[:, 1], (seg[:, 0] == 2) * 1.0)
                    val_cls_loss = cls_criterion(domain_out, val_domain_index)
                val_acc_list.append(accuracy_score(torch.max(torch.softmax(domain_out, dim=1), dim=1)[1].cpu().numpy(), val_domain_index.cpu().numpy()))
                val_loss_list.append(loss.detach().cpu().numpy())
                val_cls_loss_list.append(val_cls_loss.detach().cpu().numpy())
                output_sigmoid = torch.sigmoid(output)
                val_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
                val_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
        mean_val_loss = np.mean(val_loss_list)
        mean_val_cls_loss = np.mean(val_cls_loss_list)
        mean_val_acc = np.mean(val_acc_list)
        mean_val_disc_dice = np.mean(val_disc_dice_list)
        mean_val_cup_dice = np.mean(val_cup_dice_list)
        writer.add_scalar("Val Scalars/Val Loss", mean_val_loss, epoch)
        writer.add_scalar("Val Scalars/Val CLS Loss", mean_val_cls_loss, epoch)
        writer.add_scalar("Val Scalars/Val ACC", mean_val_acc, epoch)
        writer.add_scalar("Val Scalars/Disc Dice", mean_val_disc_dice, epoch)
        writer.add_scalar("Val Scalars/Cup Dice", mean_val_cup_dice, epoch)

        print('  Val loss: {}; CLS loss: {}\n'
              '  Val disc dice: {}; Cup dice: {}; CLS acc: {}'.format(mean_val_loss, mean_val_disc_dice, mean_val_disc_dice, mean_val_cup_dice, mean_val_acc))

        if epoch % save_interval == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format('latest'))
            torch.save(saved_model, join(model_folder, 'model_latest.model'))
        if (epoch+1) % 200 == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format(epoch+1))
            torch.save(saved_model, join(model_folder, 'model_{}.model'.format(epoch+1)))

        time_per_epoch = time() - start_epoch
        print('  Durations: {}'.format(time_per_epoch))
        writer.add_scalar("Time/Time per epoch", time_per_epoch, epoch)
    saved_model = {
        'epoch': epoch + 1 if epoch != 0 else start_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    print('Saving model_{}.model...'.format('final'))
    torch.save(saved_model, join(model_folder, 'model_final.model'))
    if isfile(join(model_folder, 'model_latest.model')):
        os.remove(join(model_folder, 'model_latest.model'))
    total_time = time() - start
    print("Running %d epochs took a total of %.2f seconds." % (num_epochs, total_time))

    # inference
    from docr.inference.inference_nets.inference_docr import inference
    for ts_csv_path in ts_csv:
        inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
        print("Running inference: {}".format(inference_tag))
        inference('model_final.model', gpu, log_folder, patch_size, root_folder, [ts_csv_path], inference_tag)

