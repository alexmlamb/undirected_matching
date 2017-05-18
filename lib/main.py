#!/usr/bin/env python
'''Main script

'''

import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger('UDGAN')

from build import (
    add_viz, build_model, compile_train, initialize_params, make_vars, set_arch,
    test, unlabeled_results)
from data import prepare_data
from exptools import make_argument_parser, setup_out_dir
from loggers import set_stream_logger
from models import conv1, conv2, conv3
from op import set_optimizer
from train import train_model
from utils import (
    config, init_tparams, srng, dropout, inverse_sigmoid, merge_images,
    sample_multinomial, print_section, update_dict_of_lists)


try:
    SLURM_NAME = os.environ["SLURM_JOB_ID"]
except:
    SLURM_NAME = 'NA'
    
ARCH = conv1

    
# MAIN -------------------------------------------------------------------------
    
_model_defaults = dict(
    num_steps=3,
    pd_steps=1,
    dim_z=128,
    loss='bgan',
    n_samples=10,
    start_on_x=False
)

_optimizer_defaults = dict(
    optimizer='rmsprop',
    op_args=dict(learning_rate=1e-4)#, beta1=0.5)
)

_data_defaults = dict(
    batch_size=64,
    pad_to=None,#(32, 32)
    tanh=True
)

_train_defaults = dict(
    num_epochs=200,
    archive_every=10
)

_visualize_defaults = dict(
    num_steps_long=4,
    visualize_every_update=0, # Not used yet
    noise_damping=0.9
)
    

def main(source, data_args, model_args, optimizer_args, train_args,
         visualize_args, binary_dir=None, image_dir=None):
    set_arch(ARCH)
    datasets, data_shapes = prepare_data(args.source, model_args=model_args,
                                         **data_args)
    inputs, k_map = make_vars(**data_shapes['train'])
    gparams, dparams = initialize_params(**model_args)
    #test(inputs, datasets, **model_args)
    d = {}
    d.update(**model_args)
    d.update(**inputs)
    model_outs = build_model(**d)
    d = {}
    d.update(**model_outs)
    d.update(**model_args)
    results = unlabeled_results(**d)
    updates = set_optimizer(results['D loss'], results['G loss'],
                            dparams, gparams, **optimizer_args)
    
    train_fn = compile_train(updates, inputs, results)
    viz1 = add_viz(viz_type='simple', Z=inputs['Z'], outputs=dict(X=model_outs['p_xs'][-1]),
                   out_dir=image_dir, archive=True, name='gen_X')
    
    viz2 = add_viz(viz_type='chain', Z=inputs['Z'], n_steps=20,
                   out_dir=image_dir, name='chain_X',
                   extra_args=dict(dim_c=model_args['dim_c'],
                                   dim_x=model_args['dim_x'],
                                   dim_y=model_args['dim_y']),
                   **model_args)
    
    viz3 = add_viz(viz_type='chain', X=inputs['X'], Z=inputs['Z'], n_steps=20,
                   out_dir=image_dir, name='inpaint_chain_X', fix_half_X=True,
                   extra_args=dict(dim_c=model_args['dim_c'],
                                   dim_x=model_args['dim_x'],
                                   dim_y=model_args['dim_y']),
                   **model_args)
    
    try:
        logger.info('Training with args {}'.format(train_args))
        train_model(train_fn, k_map, [viz1, viz2, viz3], ARCH, datasets,
                    data_shapes, model_args['dim_z'], binary_dir=binary_dir,
                    **train_args)
    except KeyboardInterrupt:
        logger.info('Training interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    
if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    set_stream_logger(args.verbosity)
    out_paths = setup_out_dir(args.out_path, args.name)
    
    data_args = {}
    data_args.update(**_data_defaults)
    
    model_args = {}
    model_args.update(**ARCH._defaults)
    model_args.update(**_model_defaults)
    
    optimizer_args = {}
    optimizer_args.update(**_optimizer_defaults)
    
    train_args = {}
    train_args.update(**_train_defaults)
    train_args['batch_size'] = data_args['batch_size']
    
    visualize_args = {}
    visualize_args.update(**_visualize_defaults)
    
    config(data_args, model_args, optimizer_args, train_args, visualize_args,
           config_file=args.config_file)
    
    main(args.source, data_args, model_args, optimizer_args, train_args,
         visualize_args, **out_paths)
    
