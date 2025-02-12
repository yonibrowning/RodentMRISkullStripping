from keras import backend as K


def dice_coef(y_true, y_pred):
    y_true_f = K.cast(K.flatten(y_true),'float32')
    y_pred_f = K.cast(K.flatten(y_pred),'float32')
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)