from sklearn.metrics import precision_recall_curve

def compute_mAP_metrics(model, dataset, num_classes=None, bounding_box_format=None, iou_th=0.5):
    from keras.utils import Progbar
    from keras.utils.io_utils import print_msg
    from keras_cv import bounding_box
    import tensorflow
    import numpy as np
    
    bounding_box_format = bounding_box_format if bounding_box_format is not None else dataset.bounding_box_format
    
    if hasattr(dataset, 'class_names'):
        class_names = dataset.class_names
        if num_classes:
            assert len(class_names) == num_classes
        else:
            num_classes = len(class_names)
    else:
        class_names = [str(_) for _ in range(num_classes)]
    
    list_classes = list()
    list_predictions = list()
    list_scores = list()
    target_format = "yxyx"

    try:
        pbar = Progbar(len(dataset))
    except:
        pbar = Progbar(None)

    for imges, boxes_gt in dataset:
        boxes_pred = model.predict(imges, verbose=False)
        boxes_pred = {k: tensorflow.convert_to_tensor(boxes_pred[k]) for k in boxes_pred}
        
        boxes_gt = bounding_box.convert_format(
            boxes_gt, source=bounding_box_format, target=target_format
        )
        boxes_pred = bounding_box.convert_format(
            boxes_pred, source=bounding_box_format, target=target_format
        )
        
        for index_0 in range(len(imges)):
            for c in range(num_classes):
              target_valid = boxes_gt["classes"][index_0]==c
              target_boxes = boxes_gt["boxes"][index_0][target_valid]
              target_valid = tensorflow.ones(len(target_boxes), tensorflow.bool)

              detection_valid = boxes_pred["classes"][index_0]==c
              detection_boxes = boxes_pred["boxes"][index_0][detection_valid]
              detection_scores = boxes_pred["confidence"][index_0][detection_valid]
              inds = tensorflow.argsort(detection_scores, direction='DESCENDING')
              iou = bounding_box.compute_iou(detection_boxes,target_boxes,
                                             bounding_box_format=target_format)
              
              for i in inds:
                  s = detection_scores[i].numpy()
                  comb = iou[i] * tensorflow.cast(target_valid, iou.dtype)
                  th = tensorflow.math.maximum(tensorflow.reduce_max(comb), iou_th)
                  comb = comb>=th
                  p = tensorflow.reduce_any(comb).numpy()
                  target_valid = target_valid & tensorflow.logical_not(comb)

                  list_classes.append(c)
                  list_predictions.append(p)
                  list_scores.append(s)
              
              inds = tensorflow.math.count_nonzero(target_valid).numpy()
              for i in range(inds):
                  list_classes.append(c)
                  list_predictions.append(True)
                  list_scores.append(np.nan)
                 
        pbar.add(1)
    
    list_classes = np.asarray(list_classes)
    list_predictions = np.asarray(list_predictions)
    list_scores = np.asarray(list_scores)
    min_score = np.nanmin(list_scores) 
    list_scores[np.isnan(list_scores)] = min_score - 1.0
    pbar.update(pbar._seen_so_far , values=None, finalize=True)

    ap = [np.nan, ] * num_classes
    for c in range(num_classes):
        valid = list_classes==c
        if np.any(list_predictions[valid]):
            precision, recall, ths = precision_recall_curve(list_predictions[valid], list_scores[valid])
            ths = np.concatenate((ths, [np.PINF,]),0)
            precision[ths<min_score] = 0
            ap[c] = -np.sum(np.diff(recall) * np.array(precision)[:-1])
            
        print_msg('AP of %30s = %7.5f' % (class_names[c], ap[c]))

    return {'mAP': np.nanmean(ap)}
