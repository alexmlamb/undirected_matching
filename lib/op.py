'''Sets optimizer.

'''

import logging

import lasagne


logger = logging.getLogger('UDGAN.optimizer')


def set_optimizer(dloss, gloss, dparams, gparams,
                  optimizer=None, op_args=None):
    '''Sets the loss and optimizer and gets updates
    
    '''
    op_args = op_args or {}
    
    logger.info("Setting optimizer. Using {} with args {}".format(
        optimizer, op_args))
    
    if optimizer == 'rmsprop':
        dupdates = lasagne.updates.rmsprop(dloss, dparams.values(), **op_args)
        gcupdates = lasagne.updates.rmsprop(gloss, gparams.values(), **op_args)
    elif optimizer == 'adam':
        dupdates = lasagne.updates.adam(dloss, dparams.values(), **op_args)
        gcupdates = lasagne.updates.adam(gloss, gparams.values(), **op_args)
    else:
        raise NotImplementedError(optimizer)

    dgcupdates = dupdates.copy()
    dgcupdates.update(gcupdates)
    
    return dgcupdates