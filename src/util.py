from data_names import *

def get_emotion_from_filename(filename):
    return split_path(filename)[-1].split('.')[0].split('_')[1]


def get_emotion_number_from_filename(filename):
    return EMOTION_NUMBERS[split_path(filename)[-1].split('.')[0].split('_')[1]]


def filename_to_category_vector(filename, category=None):
    emotion_number = get_emotion_number_from_filename(filename)
    if category is None:
        zeros = np.zeros(len(EMOTIONS), dtype='int16')
        zeros[emotion_number] = 1
        return zeros
    else:
        category_number = EMOTION_NUMBERS[category]
        zeros = np.zeros(2)
        # 0 index is negative, 1 index is positive
        zeros[int(category_number == emotion_number)] = 1
        return
