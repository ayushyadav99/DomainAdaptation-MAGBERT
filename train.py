""" Created a new file to add pretrain , adapt and train code """


"""Adversarial adaptation to train target encoder."""

import torch
import torch.nn.functional as F
import torch.nn as nn
import param
import torch.optim as optim
import os
from tqdm import tqdm, trange
from global_configs import DEVICE
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from sklearn.metrics import precision_recall_fscore_support

def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def save_model(args, net, name):
    """Save trained model."""
    folder = os.path.join(param.model_root, args.src, args.model, str(args.seed))
    path = os.path.join(folder, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(net.state_dict(), path)
    print("save pretrained model to: {}".format(path))

def pretrain(args, encoder, classifier, data_loader, optimizer, scheduler):
    """Train classifier for source domain."""

    # setup criterion and optimizer
    # optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                        #    lr=param.c_learning_rate)

    # set train state for Dropout and BN layers
    encoder.to(DEVICE)
    classifier.to(DEVICE)
    encoder.train()
    classifier.train()
    for epoch in range(args.pre_epochs):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            feat = encoder(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
            feat.to(DEVICE)
            outputs = classifier(feat)
            outputs.to(DEVICE)
            # logits = outputs[0]
            logits = outputs
            # print("mine", logits.shape, label_ids.shape)
            #print(logits, label_ids)

            #loss_fct = MSELoss()
            #loss = loss_fct(logits.view(-1), label_ids.view(-1))
            #print('logits.view(-1) : ' + str(logits.view(-1)) + "label_ids.view(-1) : " + str(label_ids.view(-1)) + "\n")

            CELoss = nn.CrossEntropyLoss()
            loss = CELoss(logits, label_ids.view(-1).long())

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            loss.backward()

            tr_loss += loss.item()
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        print("Loss so far: " +str(tr_loss/nb_tr_steps)+"\n")

    # save final model
    # save_model(args, encoder, param.src_encoder_path)
    # save_model(args, classifier, param.src_classifier_path)

    return encoder, classifier

def adapt(args, src_encoder, tgt_encoder, discriminator,
          src_classifier, src_data_loader, tgt_data_train_loader, tgt_data_all_loader):
    """Train encoder for target domain."""

    # set train state for Dropout and BN layers
    src_encoder.eval()
    src_classifier.eval()
    tgt_encoder.train()
    discriminator.train()

    # setup criterion and optimizer
    BCELoss = nn.BCELoss()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    optimizer_G = optim.Adam(tgt_encoder.parameters(), lr=param.d_learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=param.d_learning_rate)
    len_data_loader = min(len(src_data_loader), len(tgt_data_train_loader))
    
    for epoch in range(args.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_train_loader))
        print("Loader lengths: ", len(src_data_loader), len(tgt_data_train_loader))
        d_l = []
        g_l = []
        kd_l = []
        accu =  []
        for step, (batch1, batch2) in data_zip:
            batch1 = tuple(t.to(DEVICE) for t in batch1)
            batch2 = tuple(t.to(DEVICE) for t in batch2)
            src_input_ids, src_visual, src_acoustic, src_input_mask, src_segment_ids, src_label_ids = batch1 
            tgt_input_ids, tgt_visual, tgt_acoustic, tgt_input_mask, tgt_segment_ids, tgt_label_ids = batch2

            src_visual = torch.squeeze(src_visual, 1)
            src_acoustic = torch.squeeze(src_acoustic, 1)
            # zero gradients for optimizer
            optimizer_D.zero_grad()

            # extract and concat features
            with torch.no_grad():
                feat_src = src_encoder(src_input_ids,
                    src_visual,
                    src_acoustic,
                    token_type_ids=src_segment_ids,
                    attention_mask=src_input_mask,
                    labels=None,)

            feat_src_tgt = tgt_encoder(src_input_ids,
                    src_visual,
                    src_acoustic,
                    token_type_ids=src_segment_ids,
                    attention_mask=src_input_mask,
                    labels=None,)
            feat_tgt = tgt_encoder(tgt_input_ids,
                    tgt_visual,
                    tgt_acoustic,
                    token_type_ids=tgt_segment_ids,
                    attention_mask=tgt_input_mask,
                    labels=None,)

            feat_concat = torch.cat((feat_src_tgt, feat_tgt), 0)
            # print("Checking dims", feat_src_tgt.shape, feat_tgt.shape, feat_concat.shape)

            # predict on discriminator
            pred_concat = discriminator(feat_concat.detach())
            pred_concat.to(DEVICE)

            # prepare real and fake label
            label_src = make_cuda(torch.ones(feat_src_tgt.size(0))).unsqueeze(1)
            label_tgt = make_cuda(torch.zeros(feat_tgt.size(0))).unsqueeze(1)
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for discriminator
            dis_loss = BCELoss(pred_concat, label_concat)
            dis_loss.backward()

            for p in discriminator.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)
            # optimize discriminator
            optimizer_D.step()
            # print("pred_concat : " + str(pred_concat) + "\n")
            pred_cls = (pred_concat>0.5).float()
            # pred_cls = torch.squeeze(pred_concat.max(1)[1])
            # print("pred_cls : " + str(pred_cls) + "\n")
            # print("label Concat: " + str(label_concat) + "\n")
            acc = (pred_cls == label_concat).float().mean()

            # zero gradients for optimizer
            optimizer_G.zero_grad()
            T = args.temperature

            # predict on discriminator
            pred_tgt = discriminator(feat_tgt)

            # logits for KL-divergence
            with torch.no_grad():
                src_prob = F.softmax(src_classifier(feat_src) / T, dim=-1)
            tgt_prob = F.log_softmax(src_classifier(feat_src_tgt) / T, dim=-1)
            kd_loss = KLDivLoss(tgt_prob, src_prob.detach()) * T * T

            # compute loss for target encoder
            label_src_n = make_cuda(torch.ones(feat_tgt.size(0))).unsqueeze(1)
            # print("Checking dims 2:", feat_tgt.shape, pred_tgt.shape, label_src.shape)
            gen_loss = BCELoss(pred_tgt, label_src_n)
            loss_tgt = args.alpha * gen_loss + args.beta * kd_loss
            loss_tgt.backward()
            accu.append(acc.item())
            g_l.append(gen_loss.item())
            d_l.append(dis_loss.item())
            kd_l.append(kd_loss.item())
            torch.nn.utils.clip_grad_norm_(tgt_encoder.parameters(), args.max_grad_norm)
            # optimize target encoder
            optimizer_G.step()

            if (step + 1) % args.log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: "
                      "acc=%.4f g_loss=%.4f d_loss=%.4f kd_loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len_data_loader,
                         torch.tensor(accu).mean(),
                         torch.tensor(g_l).mean(),
                         torch.tensor(d_l).mean(),
                         torch.tensor(kd_l).mean()))

        evaluate(args, tgt_encoder, src_classifier, tgt_data_all_loader)

    return tgt_encoder

def evaluate(args, encoder, classifier, data_loader):
    encoder.eval()
    classifier.eval()
    y_true = []
    y_pred = []
    dev_loss = 0
    acc = 0
    num_acc = 0
    nb_dev_examples, nb_dev_steps = 0, 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            feat = encoder(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
            outputs = classifier(feat)
            logits = outputs


            #loss_fct = MSELoss()
            #loss = loss_fct(logits.view(-1), label_ids.view(-1))
            CELoss = nn.CrossEntropyLoss()
            loss = CELoss(logits, label_ids.view(-1).long())
            pred_cls = logits.data.max(1)[1]
            y_pred.extend(pred_cls.cpu().tolist())
            y_true.extend(label_ids.view(-1).cpu().tolist())
            #print(pred_cls.cpu()) 
            #print(label_ids.view(-1).cpu())
            # acc += pred_cls.eq(logits.data).cpu().sum().item()

            #print("logits:" + str(logits))
            #print("acc:" + str(acc))
            #print("pred_cls: " + str(pred_cls))
            #print("labels_ids.view(-1):" + str(label_ids.view(-1)))
            #print("labels_ids:" + str(label_ids))

            acc += (pred_cls.eq(label_ids.view(-1)).cpu().sum().item())
            num_acc += len(pred_cls)
            # print("num_acc", num_acc)

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1
        
        #print(y_true)
        #print(y_pred) 
        
        print(precision_recall_fscore_support(y_true, y_pred, average=None))
        print("Avg Loss = %.4f, Avg Accuracy = %.4f" % (dev_loss/nb_dev_steps, acc/num_acc))
    return dev_loss / nb_dev_steps
