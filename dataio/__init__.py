def get_data(args, return_val=False, val_downscale=4.0, **overwrite_cfgs):
    dataset_type = args.data.get('type', 'DTU')
    cfgs = {
        'scale_radius': args.data.get('scale_radius', -1),
        'downscale': args.data.downscale,
        'data_dir': args.data.data_dir,
        'train_cameras': False
    }
    
    if dataset_type == 'DTU':
        from .DTU import SceneDataset
        cfgs['cam_file'] = args.data.get('cam_file', None)
    elif dataset_type == "TUM":
        from .TUM import SceneDataset
        cfgs['time_downsample_factor'] = args.data.get('time_downsample_factor', 24)
        cfgs['radius_init'] = args.model.surface.get('radius_init', 1.)
        cfgs['start_moving'] = args.data.get('start_moving', -1)
        cfgs['normalize_mode'] = args.model.get('normalize_mode', "shift_and_scale")
    elif dataset_type == 'custom':
        from .custom import SceneDataset
    elif dataset_type == 'BlendedMVS':
        from .BlendedMVS import SceneDataset
    else:
        raise NotImplementedError

    cfgs.update(overwrite_cfgs)
    dataset = SceneDataset(**cfgs)
    if return_val:
        cfgs['downscale'] = val_downscale
        val_dataset = SceneDataset(**cfgs)
        return dataset, val_dataset
    else:
        return dataset