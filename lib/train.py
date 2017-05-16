'''Main train function.

'''

import logging
import sys
import time

import numpy as np
import numpy.random as rng
from progressbar import Bar, ProgressBar, Percentage, Timer
import theano

from utils import print_section, save_parameters, update_dict_of_lists


logger = logging.getLogger('UDGAN.train')
floatX = theano.config.floatX


def train_model(train_fn, k_map, vizs, arch, datasets, data_shapes, dim_z,
                num_epochs=None, batch_size=None,
                archive_every=None, binary_dir=None):
    '''Train method.
    
    '''
    
    train_samples = data_shapes['train']['features'][0]
    z_out_p = None
    results = {}
    
    for epoch in xrange(num_epochs):
        u = 0
        start_time = time.time()
        widgets = ['Epoch {}, '.format(epoch), Timer(), Bar()]
        pbar = ProgressBar(
            widgets=widgets, maxval=(train_samples // batch_size)).start()
        
        e_results = {}
        iterator = datasets['train'].get_epoch_iterator(as_dict=True)
        x_in_ = None
        label_ = None
        
        for batch in iterator:
            inputs = [batch[k_map[k]] for k in ['X', 'Y']]
            if batch_size is None: batch_size = batch[0].shape[0]
            if inputs[0].shape[0] != batch_size: break
                        
            z_in = rng.normal(size=(batch_size, dim_z)).astype(floatX)
            inputs = tuple(inputs + [z_in])
            outs = train_fn(*inputs)
            if np.any(np.array([np.isnan(o_) for o_ in outs.values()])):
                logger.error('NaN found')
                logger.error(outs)
                exit(0)
            update_dict_of_lists(e_results, **outs)
            u += 1
            pbar.update(u)
            
        # RESULTS
        e_results = dict((k, np.mean(v)) for k, v in e_results.items())
        update_dict_of_lists(results, **e_results)
        
        print_section('Epoch {} completed')
        logger.info('Epoch {} of {} took {:.3f}s'.format(
            epoch + 1, num_epochs, time.time() - start_time))
        logger.info(e_results)
        
        # SAVE
        if archive_every is not None and epoch % archive_every:
            suffix = '_{}'.format(epoch)
        else:
            suffix = ''
        logger.debug('Saving to {}'.format(binary_dir))
        save_parameters(arch, out_path=binary_dir, suffix=suffix)
        
        # Images
        iterator = datasets['test'].get_epoch_iterator(as_dict=True)
        inputs = iterator.next()
        inputs = dict((k, inputs[k_map[k]]) for k in ['X', 'Y'])
        inputs['Z'] = rng.normal(size=(inputs['X'].shape[0], dim_z)).astype(floatX)
        for viz in vizs:
            if viz.archive:
                suffix = epoch
            else:
                suffix = ''
            viz.run(inputs, suffix=suffix)
        
