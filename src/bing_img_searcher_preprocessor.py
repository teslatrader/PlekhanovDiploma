from bing_image_downloader.downloader import download
import os
from men_to_load import men_to_load as mens_list
from women_to_load import women_to_load as womens_list
from PIL import Image
import tensorflow as tf


def imgs_list_update(path_to_list: str, list_of_imgs: list):
    for idx, image in enumerate(list_of_imgs):
        list_of_imgs[idx] = f'{path_to_list}/{image}'
    return list_of_imgs


def image_prepocessing(imgs_list: list, current_dir: str, side_size=256):
    idx_flipped = len(imgs_list) + 1
    for image in imgs_list:
        try:
            img = Image.open(image)
            # img.show()
            img = img.convert('RGB')

            # normlization and resize to single side size (by bigger side)
            size = img.size
            value_max = max(size)
            normalization_ratio = round(value_max / side_size, 2)
            width_normalized = int(size[0] / normalization_ratio)
            height_normalized = int(size[1] / normalization_ratio)
            img = tf.cast(img, tf.float32)
            img = tf.image.resize(img, [height_normalized, width_normalized])

            # augmentation - flip the image
            flipped = tf.image.flip_left_right(img)
            flipped_name = f'{current_dir}/Image_{idx_flipped}.jpg'

            # save original and augumentated images
            tf.keras.utils.save_img(image, img)
            tf.keras.utils.save_img(flipped_name, flipped)
            idx_flipped += 1

        except Exception as e:
            print(f'During image normalization we got exception: \n{e}')
            os.remove(image)


def get_images(
        list_to_load: list,
        quantity_to_load=20,
        adult_filter_off=True,
        connection_timeout=10,
        filter_of_images='photo',
        verbose=True
):
    for person in list_to_load:
        try:
            query = f'face of {person}'
            path_to_save = f'../inputs/'
            download(
                query,
                limit=quantity_to_load,
                output_dir=path_to_save,
                adult_filter_off=adult_filter_off,
                timeout=connection_timeout,
                filter=filter_of_images,
                verbose=verbose
            )

            # adjust name of source folders
            path_src = f'{path_to_save}/{query}'
            path_dest = f'{path_to_save}/{person}'
            os.rename(path_src, path_dest)

            # creating list of downloaded images and execute preprocessing with each of image
            imgs_list = os.listdir(path_dest)
            # print(imgs_list)
            imgs_list = imgs_list_update(path_dest, imgs_list)
            image_prepocessing(imgs_list, path_dest)
        except Exception as e:
            print(f'During image loading, we got exception: \n{e}')


if __name__ == '__main__':
    get_images(mens_list)
    get_images(womens_list)
