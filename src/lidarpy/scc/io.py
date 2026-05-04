import os
import shutil
from loguru import logger


def move2odir(source_filename, destination, overwrite=True):
    """
    It moves the file named source_filename in local directory to the absolute path file 'moved_filepath'.
    Input:
    source_filename: full path filename (string). 
    destination: destination path (string). 
    Output:
    sent: control variable. True=sent; False=not sent. (boolean).
    """
    sent = False    
    logger.info('Moving %s to %s' % (source_filename, destination))
    try:
        # create directory if it does not exist
        if not os.path.exists(destination):
            os.makedirs(destination)

        # delete file if exists in destination
        if os.path.exists(os.path.join(destination, os.path.basename(source_filename))):
            os.remove(os.path.join(destination, os.path.basename(source_filename)))

        # copyfile (shutil da problemas de incoherencia entre ejecucion (bien) y mensaje (error))
        sent = shutil.copy(source_filename, destination)
        if sent:
            sent = True
        logger.info('Moving file %s to user-defined output directory: %s... DONE!' % (source_filename, destination))

        # delete original
        os.remove(source_filename)
    except:
        if os.path.exists(os.path.join(destination, os.path.basename(source_filename))):
            logger.info('Moving file %s to user-defined output directory: %s... DONE!' % (source_filename, destination))
            sent = True
            # delete original
            os.remove(source_filename)
        else:
            logger.warning('Moving file %s to user-defined output directory: %s... ERROR!' % (source_filename, destination))
    return sent