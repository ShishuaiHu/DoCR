from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from torch.cuda.amp import autocast
from torch.utils import data
from docr.utils.file_utils import check_folders
from batchgenerators.utilities.file_and_folder_operations import *
from docr.models.DoCR import DoCR
from docr.datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from docr.datasets.utils.convert_csv_to_list import convert_labeled_list
from docr.datasets.utils.transform import collate_fn_ts
from docr.utils.metrics.dice import get_hard_dice
from docr.utils.visualization import visualization_as_nii
from docr.training.train_nets.train_docr import encoding_domain
from sklearn.metrics import accuracy_score
from docr.utils.fourier import FDA_img_to_hfi


def inference(chk_name, gpu, log_folder, patch_size, root_folder, ts_csv, inference_tag='all'):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu])
    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    visualization_folder = join(visualization_folder, inference_tag)
    maybe_mkdir_p(visualization_folder)

    ts_img_list, ts_label_list = convert_labeled_list(ts_csv, r=1)
    if ts_label_list is None:
        evaluate = False
        ts_dataset = RIGA_unlabeled_set(root_folder, ts_img_list, patch_size)
    else:
        evaluate = True
        ts_dataset = RIGA_labeled_set(root_folder, ts_img_list, ts_label_list, patch_size)
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=4,
                                                num_workers=2,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=collate_fn_ts)
    model = DoCR()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    assert isfile(join(model_folder, chk_name)), 'missing model checkpoint {}!'.format(join(model_folder, chk_name))
    params = torch.load(join(model_folder, chk_name))
    model.load_state_dict(params['model_state_dict'])

    seg_list = list()
    output_list = list()
    data_list = list()
    cls_pro_list = list()
    cls_list = list()
    cls_true = list()
    rec_list = list()
    hf_list = list()
    with torch.no_grad():
        model.eval()
        for iter, batch in enumerate(ts_dataloader):
            data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
            hf = FDA_img_to_hfi(batch['data'], L=0.01)
            domain_encoding = encoding_domain(root_folder, batch['name'])
            domain_index = domain_encoding.argmax(dim=1)
            with autocast():
                cls_out = model.forward_cls(data)
                output, rec_output = model(data, domain_label=domain_encoding, training=True)
            output_sigmoid = torch.sigmoid(output).cpu().numpy()
            seg_list.append(batch['seg'])
            output_list.append(output_sigmoid)
            data_list.append(batch['data'])
            rec_out_tanh = torch.tanh(rec_output)
            rec_list.append(rec_out_tanh.cpu().numpy())
            hf_list.append(hf)

            cls_out = cls_out.to(dtype=torch.float32)
            cls_pro = torch.softmax(cls_out, dim=1)
            cls_pro_list.append(cls_pro.cpu().numpy())
            _, cls_out_softmax = torch.max(cls_pro, dim=1)
            cls_list.append(cls_out_softmax.cpu().numpy())
            cls_true.append(domain_index.cpu().numpy())
    all_data = list()
    all_seg = list()
    all_output = list()
    all_cls_out = list()
    all_cls_pro_out = list()
    all_cls_true = list()
    all_rec = list()
    all_hf = list()
    for i in range(len(data_list)):
        for j in range(data_list[i].shape[0]):
            all_data.append(data_list[i][j])
            all_seg.append(seg_list[i][j])
            all_output.append(output_list[i][j])
            all_cls_out.append(cls_list[i][j])
            all_cls_pro_out.append(cls_pro_list[i][j])
            all_cls_true.append(cls_true[i][j])
            all_rec.append(rec_list[i][j])
            all_hf.append(hf_list[i][j])
    all_data = np.stack(all_data)
    all_seg = np.stack(all_seg)
    all_output = np.stack(all_output)
    all_rec = np.stack(all_rec)
    all_hf = np.stack(all_hf)
    visualization_as_nii(all_data[:, 0].astype(np.float32), join(visualization_folder, 'data_channel0.nii.gz'))
    visualization_as_nii(all_data[:, 1].astype(np.float32), join(visualization_folder, 'data_channel1.nii.gz'))
    visualization_as_nii(all_data[:, 2].astype(np.float32), join(visualization_folder, 'data_channel2.nii.gz'))
    visualization_as_nii(all_output[:, 0].astype(np.float32), join(visualization_folder, 'output_disc.nii.gz'))
    visualization_as_nii(all_output[:, 1].astype(np.float32), join(visualization_folder, 'output_cup.nii.gz'))
    visualization_as_nii(all_rec[:, 0].astype(np.float32), join(visualization_folder, 'rec_channel0.nii.gz'))
    visualization_as_nii(all_rec[:, 1].astype(np.float32), join(visualization_folder, 'rec_channel1.nii.gz'))
    visualization_as_nii(all_rec[:, 2].astype(np.float32), join(visualization_folder, 'rec_channel2.nii.gz'))
    visualization_as_nii(all_hf[:, 0].astype(np.float32), join(visualization_folder, 'hf_channel0.nii.gz'))
    visualization_as_nii(all_hf[:, 1].astype(np.float32), join(visualization_folder, 'hf_channel1.nii.gz'))
    visualization_as_nii(all_hf[:, 2].astype(np.float32), join(visualization_folder, 'hf_channel2.nii.gz'))
    acc = accuracy_score(np.array(all_cls_true), np.array(all_cls_out))
    if evaluate:
        visualization_as_nii(all_seg[:, 0].astype(np.float32), join(visualization_folder, 'seg.nii.gz'))
        disc_dice = get_hard_dice(torch.from_numpy(all_output[:, 0]), torch.from_numpy(((all_seg[:, 0] > 0) * 1.0)))
        cup_dice = get_hard_dice(torch.from_numpy(all_output[:, 1]), torch.from_numpy(((all_seg[:, 0] > 1) * 1.0)))
        metrics_str = 'Tag: {}\n  Disc dice: {}; Cup dice: {}; Acc: {}.'.format(inference_tag, disc_dice, cup_dice, acc)
        print(metrics_str)
        with open(join(metrics_folder, '{}.txt'.format(inference_tag)), 'w') as f:
            f.write(metrics_str)
    with open(join(visualization_folder, 'cls.csv'), 'w') as f:
        f.write('case,prediction,probability,true\n')
        for i in range(len(all_cls_out)):
            f.write('{},{},{},{}\n'.format(i, all_cls_out[i], all_cls_pro_out[i],all_cls_true[i]))
